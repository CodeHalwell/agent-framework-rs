//! A Redis [`VectorStore`]: HASH or JSON documents indexed by Redis Search.
//!
//! Port of upstream `agent_framework_redis._vector_store` (`RedisCollection`,
//! `RedisStore`). Upstream drives RedisVL; this module issues the `FT.*`,
//! `HSET`/`HMGET` and `JSON.*` commands itself over the crate's async client.
//!
//! Requires Redis Search with `INDEXMISSING`/`INDEXEMPTY` support, and
//! RedisJSON for JSON storage — Redis 8 ships both (upstream tests against
//! 8.0.3+). Redis Search indexes database 0 only, so a URL selecting another
//! database is refused.
//!
//! ```no_run
//! use agent_framework_core::vectors::{
//!     Filter, VectorSearchOptions, VectorStore, VectorStoreCollectionDefinition, VectorStoreField,
//! };
//! use agent_framework_redis::RedisVectorStore;
//!
//! # async fn demo() -> agent_framework_core::error::Result<()> {
//! let store = RedisVectorStore::new("redis://localhost:6379")?;
//! let definition = VectorStoreCollectionDefinition::new(vec![
//!     VectorStoreField::key("id"),
//!     VectorStoreField::data("year").with_type("int").indexed(),
//!     VectorStoreField::vector("embedding", 4),
//! ])?;
//! let docs = store.get_collection("docs", definition)?;
//! docs.ensure_collection_exists().await?;
//! let hits = docs
//!     .search(
//!         vec![0.1, 0.2, 0.3, 0.4],
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
//! - Names are namespaced: collection `c` in namespace `n` is the index
//!   `af:vector:{hex(n)}:index:{hex(c)}` over documents prefixed
//!   `af:vector:{hex(n)}:data:{hex(c)}:`, so arbitrary names cannot collide
//!   with each other or with foreign keys.
//!   [`list_collection_names`](VectorStore::list_collection_names) decodes only canonical names.
//! - Keys are non-empty strings. Storage names must be ASCII identifiers of
//!   at most 128 characters and must not start with `_af_`.
//! - Indexed `int`/`float` fields are `NUMERIC`; indexed `str`, `bool`,
//!   `list` and `tuple` fields are case-sensitive `TAG`s separated by
//!   U+001F. The key field is always indexed. Only indexed fields can be
//!   filtered.
//! - HASH documents store vectors as little-endian binary, booleans as
//!   `true`/`false`, unindexed containers as JSON text and indexed string
//!   lists joined by U+001F. HASH has no null: a null data value is refused
//!   (use JSON storage for nullable fields). JSON documents store values
//!   natively.
//! - Scores are native distances, lower is closer, reported with
//!   [`VectorSearchResult::SCORE_KIND_DISTANCE`]: `DEFAULT` and
//!   `cosine_distance` search with `COSINE`, `euclidean_squared_distance`
//!   with `L2`, and `redis.ip` with `IP` (1 − dot product).
//! - [`VectorSearchOptions::provider_filter`] is a Redis Search query
//!   fragment, conjoined with the portable filter.
//!
//! # Divergences
//!
//! - Upstream's `score_threshold` (a `VECTOR_RANGE` query) has no
//!   counterpart: the Rust search options carry no threshold.
//! - Filtered retrieval (`get` with a filter and `order_by`) is not part of
//!   the Rust [`VectorCollection`] surface; `get` takes keys.
//! - The `bytes` data type is refused: records are JSON, which has no bytes.
//!   A `bytes` vector type is accepted and stored as `float32`, as upstream.

use std::sync::Arc;

use agent_framework_core::error::{Error, Result};
use agent_framework_core::vectors::{
    FieldType, Filter, FilterExpression, FilterGroupOperator, FilterOperator, VectorCollection,
    VectorSearchOptions, VectorSearchResult, VectorStore, VectorStoreCollectionDefinition,
    VectorStoreField,
};
use async_trait::async_trait;
use redis::Value as RedisValue;
use serde_json::{json, Map, Value};

use crate::internal::{map_redis_err, LazyConnection};

/// Records per pipeline.
const BATCH_SIZE: usize = 100;
/// The TAG separator, as upstream: a control character no ordinary string
/// contains.
pub const TAG_SEPARATOR: char = '\u{1f}';
/// The attribute KNN and range searches yield their distance under.
const DISTANCE_FIELD: &str = "_af_distance";

/// How documents are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RedisStorageType {
    /// Redis hashes, vectors as binary (the default).
    #[default]
    Hash,
    /// RedisJSON documents, vectors as arrays.
    Json,
}

impl RedisStorageType {
    fn as_str(self) -> &'static str {
        match self {
            Self::Hash => "HASH",
            Self::Json => "JSON",
        }
    }
}

fn redis_metric(function: Option<&str>) -> Option<&'static str> {
    Some(match function.unwrap_or("DEFAULT") {
        "DEFAULT" | "cosine_distance" => "COSINE",
        "euclidean_squared_distance" => "L2",
        "redis.ip" => "IP",
        _ => return None,
    })
}

fn is_float64(field: &VectorStoreField) -> bool {
    field.type_.as_deref() == Some("float64")
}

fn is_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
        && name.len() <= 128
}

fn namespace_component(value: &str) -> Result<String> {
    if value.is_empty() || value.len() > 256 {
        return Err(Error::Configuration(
            "Redis namespace and collection names must contain 1 to 256 UTF-8 bytes.".into(),
        ));
    }
    Ok(value.bytes().map(|b| format!("{b:02x}")).collect())
}

/// The index name and document prefix for a collection in a namespace.
pub fn collection_names(namespace: &str, collection: &str) -> Result<(String, String)> {
    let base = format!("af:vector:{}:", namespace_component(namespace)?);
    let collection = namespace_component(collection)?;
    Ok((
        format!("{base}index:{collection}"),
        format!("{base}data:{collection}:"),
    ))
}

/// The collection an index name in `namespace` belongs to, or `None` for a
/// foreign or non-canonical name.
pub fn parse_index_name(namespace: &str, index_name: &str) -> Option<String> {
    let prefix = format!("af:vector:{}:index:", namespace_component(namespace).ok()?);
    let encoded = index_name.strip_prefix(&prefix)?;
    if encoded.len() < 2 || encoded.len() > 512 || encoded.len() % 2 != 0 {
        return None;
    }
    let bytes = (0..encoded.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&encoded[i..i + 2], 16).ok())
        .collect::<Option<Vec<u8>>>()?;
    let name = String::from_utf8(bytes).ok()?;
    let (canonical, _) = collection_names(namespace, &name).ok()?;
    (canonical == index_name).then_some(name)
}

/// Escape a TAG term: every non-word character gets a backslash, as
/// RedisVL's `TokenEscaper`.
pub fn escape_tag(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for c in value.chars() {
        if !(c.is_alphanumeric() || c == '_') {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

fn numeric(value: &Value) -> Result<f64> {
    let Value::Number(n) = value else {
        return Err(Error::Configuration(
            "Redis numeric filters and indexed values require numbers, not booleans.".into(),
        ));
    };
    let f = n.as_f64().unwrap_or(f64::NAN);
    const LIMIT: u64 = 1 << 53;
    let integer_too_large = match (n.as_i64(), n.as_u64()) {
        (Some(i), _) => i.unsigned_abs() > LIMIT,
        (None, Some(_)) => true,
        _ => false,
    };
    if !f.is_finite() || integer_too_large {
        return Err(Error::Configuration(
            "Redis indexed numbers must be finite; integers must be within [-2**53, 2**53].".into(),
        ));
    }
    Ok(f)
}

fn tag_value(value: &Value) -> Result<&str> {
    let Value::String(s) = value else {
        return Err(Error::Configuration(
            "Redis string TAG values must be strings.".into(),
        ));
    };
    if s.trim() != s || s.contains(TAG_SEPARATOR) || s.contains('\0') {
        return Err(Error::Configuration(
            "Indexed Redis strings cannot have surrounding whitespace, NUL, or the TAG separator U+001F."
                .into(),
        ));
    }
    if s.len() > 4096 {
        return Err(Error::Configuration(
            "Redis TAG values cannot exceed the native 4096-byte term limit.".into(),
        ));
    }
    Ok(s)
}

fn tag_filter(name: &str, value: &str) -> String {
    // An empty TAG needs the explicit empty term INDEXEMPTY indexes.
    if value.is_empty() {
        format!("@{name}:{{\"\"}}")
    } else {
        format!("@{name}:{{{}}}", escape_tag(value))
    }
}

fn and(parts: &[String]) -> String {
    if parts.is_empty() {
        "*".into()
    } else {
        format!("({})", parts.join(" "))
    }
}

fn or(parts: &[String]) -> Result<String> {
    if parts.is_empty() {
        return Err(Error::Configuration(
            "Redis OR expressions require at least one operand.".into(),
        ));
    }
    Ok(format!("({})", parts.join(" | ")))
}

fn not(part: &str) -> String {
    format!("(-({part}))")
}

fn nothing(name: &str) -> String {
    format!("(ismissing(@{name}) -ismissing(@{name}))")
}

fn kind_of(field: &VectorStoreField) -> &str {
    field.type_.as_deref().unwrap_or("str")
}

fn non_null(name: &str, kind: &str, storage: RedisStorageType) -> Result<String> {
    if storage == RedisStorageType::Hash {
        return Ok(not(&format!("ismissing(@{name})")));
    }
    match kind {
        "int" | "float" => Ok(format!("@{name}:[-inf +inf]")),
        "bool" => Ok(format!("@{name}:{{true|false}}")),
        _ => Err(Error::Configuration(
            "Redis JSON TAG indexes cannot distinguish null from all non-null string/collection \
             values without enumerating terms; this operator is unsupported."
                .into(),
        )),
    }
}

fn equality(name: &str, kind: &str, value: &Value) -> Result<String> {
    match (kind, value) {
        (_, Value::Null) => Err(Error::Configuration(
            "Use is_null/is_not_null instead of equality with null.".into(),
        )),
        ("list" | "tuple", _) => Err(Error::Configuration(
            "Redis supports string collection membership, not whole-collection equality.".into(),
        )),
        ("int" | "float", Value::Number(_)) => {
            let n = numeric(value)?;
            Ok(format!("@{name}:[{n} {n}]"))
        }
        ("int" | "float", _) => Ok(nothing(name)),
        ("bool", Value::Bool(b)) => Ok(tag_filter(name, if *b { "true" } else { "false" })),
        ("bool", _) => Ok(nothing(name)),
        (_, Value::String(_)) => Ok(tag_filter(name, tag_value(value)?)),
        _ => Ok(nothing(name)),
    }
}

/// Translate one portable condition on an indexed field into Redis Search
/// query syntax.
pub fn filter_condition(
    filter: &Filter,
    field: &VectorStoreField,
    storage: RedisStorageType,
) -> Result<String> {
    let name = field.effective_storage_name();
    let kind = kind_of(field);
    let value = filter.value.as_ref().unwrap_or(&Value::Null);
    let present = not(&format!("ismissing(@{name})"));
    let items = || value.as_array().cloned().unwrap_or_default();
    Ok(match &filter.operator {
        FilterOperator::Exists => present,
        FilterOperator::IsNotNull => non_null(name, kind, storage)?,
        FilterOperator::IsNull => and(&[present, not(&non_null(name, kind, storage)?)]),
        FilterOperator::Eq => equality(name, kind, value)?,
        FilterOperator::Ne => and(&[present, not(&equality(name, kind, value)?)]),
        op @ (FilterOperator::In | FilterOperator::NotIn) => {
            // Membership never matches an actual null, even with null listed.
            let parts = items()
                .iter()
                .filter(|i| !i.is_null())
                .map(|i| equality(name, kind, i))
                .collect::<Result<Vec<_>>>()?;
            let matches = if parts.is_empty() {
                nothing(name)
            } else {
                or(&parts)?
            };
            if *op == FilterOperator::In {
                matches
            } else {
                and(&[non_null(name, kind, storage)?, not(&matches)])
            }
        }
        op @ (FilterOperator::Gt
        | FilterOperator::Gte
        | FilterOperator::Lt
        | FilterOperator::Lte
        | FilterOperator::Between) => {
            if !matches!(kind, "int" | "float") {
                return Err(Error::Configuration(
                    "Redis ordered comparisons require a numeric indexed field.".into(),
                ));
            }
            match op {
                FilterOperator::Between => {
                    let bounds = items();
                    if bounds.len() != 2 {
                        return Err(Error::Configuration("between takes two bounds".into()));
                    }
                    format!(
                        "@{name}:[{} {}]",
                        numeric(&bounds[0])?,
                        numeric(&bounds[1])?
                    )
                }
                FilterOperator::Gt => format!("@{name}:[({} +inf]", numeric(value)?),
                FilterOperator::Gte => format!("@{name}:[{} +inf]", numeric(value)?),
                FilterOperator::Lt => format!("@{name}:[-inf ({}]", numeric(value)?),
                _ => format!("@{name}:[-inf {}]", numeric(value)?),
            }
        }
        op @ (FilterOperator::Contains
        | FilterOperator::ContainsAny
        | FilterOperator::ContainsAll) => {
            if !matches!(kind, "list" | "tuple") {
                return Err(Error::Configuration(
                    "Redis collection membership requires an indexed list or tuple of strings."
                        .into(),
                ));
            }
            let values = if *op == FilterOperator::Contains {
                vec![value.clone()]
            } else {
                items()
            };
            if values.is_empty() && *op == FilterOperator::ContainsAll {
                return Err(Error::Configuration(
                    "Redis cannot distinguish null from an empty TAG array for contains_all=[]."
                        .into(),
                ));
            }
            let parts = values
                .iter()
                .map(|v| Ok(tag_filter(name, tag_value(v)?)))
                .collect::<Result<Vec<_>>>()?;
            if parts.is_empty() {
                nothing(name)
            } else if *op == FilterOperator::ContainsAll {
                and(&parts)
            } else {
                or(&parts)?
            }
        }
        other => {
            return Err(Error::Configuration(format!(
                "Redis does not support portable operator '{other:?}'; TEXT tokenization is not \
                 literal string matching."
            )))
        }
    })
}

/// Little-endian bytes of a vector in the field's datatype, after checking
/// its width, finiteness and (for cosine) magnitude.
pub fn vector_bytes(values: &[f64], field: &VectorStoreField) -> Result<Vec<u8>> {
    let dims = field
        .dimensions
        .ok_or_else(|| Error::Configuration("Redis vector fields require dimensions.".into()))?;
    if values.len() != dims {
        return Err(Error::Configuration(format!(
            "Vector '{}' must contain exactly {dims} dimensions.",
            field.name
        )));
    }
    let finite = if is_float64(field) {
        values.iter().all(|v| v.is_finite())
    } else {
        values.iter().all(|v| (*v as f32).is_finite())
    };
    if !finite {
        return Err(Error::Configuration(format!(
            "Vector '{}' must contain only finite values representable in its datatype.",
            field.name
        )));
    }
    let metric = redis_metric(field.distance_function.as_ref().map(|d| d.as_str()));
    if metric == Some("COSINE") && values.iter().all(|v| *v == 0.0) {
        return Err(Error::Configuration(format!(
            "Cosine vector '{}' must have a nonzero magnitude.",
            field.name
        )));
    }
    Ok(if is_float64(field) {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    } else {
        values
            .iter()
            .flat_map(|v| (*v as f32).to_le_bytes())
            .collect()
    })
}

fn vector_values(value: &Value, field: &VectorStoreField) -> Result<Vec<f64>> {
    let Value::Array(items) = value else {
        return Err(Error::Configuration(format!(
            "Vector '{}' must be an array of numbers.",
            field.name
        )));
    };
    items
        .iter()
        .map(|i| {
            i.as_f64().ok_or_else(|| {
                Error::Configuration(format!(
                    "Vector '{}' must contain numbers, not booleans or strings.",
                    field.name
                ))
            })
        })
        .collect()
}

fn decode_vector_bytes(bytes: &[u8], field: &VectorStoreField) -> Result<Value> {
    let width = if is_float64(field) { 8 } else { 4 };
    let dims = field.dimensions.unwrap_or_default();
    if bytes.len() != dims * width {
        return Err(Error::Service(format!(
            "Binary vector '{}' has an invalid byte length.",
            field.name
        )));
    }
    let values: Vec<f64> = bytes
        .chunks_exact(width)
        .map(|c| {
            if width == 8 {
                f64::from_le_bytes(c.try_into().expect("8 bytes"))
            } else {
                f32::from_le_bytes(c.try_into().expect("4 bytes")) as f64
            }
        })
        .collect();
    Ok(json!(values))
}

fn json_finite(value: &Value) -> Result<()> {
    match value {
        Value::Number(n) if n.as_f64().is_some_and(|f| !f.is_finite()) => Err(
            Error::Configuration("Redis JSON data must contain finite numbers.".into()),
        ),
        Value::Array(items) => items.iter().try_for_each(json_finite),
        Value::Object(map) => map.values().try_for_each(json_finite),
        _ => Ok(()),
    }
}

/// A value serialized for a HASH field: binary for vectors, text otherwise.
#[derive(Debug, Clone, PartialEq)]
pub enum HashValue {
    /// A text value.
    Text(String),
    /// A binary vector.
    Bytes(Vec<u8>),
}

/// A collection in a Redis server. See the [module docs](self).
#[derive(Clone)]
pub struct RedisVectorCollection {
    conn: Arc<LazyConnection>,
    name: String,
    definition: VectorStoreCollectionDefinition,
    storage: RedisStorageType,
    index_name: String,
    key_prefix: String,
}

impl std::fmt::Debug for RedisVectorCollection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedisVectorCollection")
            .field("name", &self.name)
            .field("storage", &self.storage)
            .field("index_name", &self.index_name)
            .finish_non_exhaustive()
    }
}

impl RedisVectorCollection {
    fn new(
        conn: Arc<LazyConnection>,
        namespace: &str,
        name: &str,
        definition: VectorStoreCollectionDefinition,
        storage: RedisStorageType,
    ) -> Result<Self> {
        for field in definition.fields() {
            let storage_name = field.effective_storage_name();
            if !is_identifier(storage_name) || storage_name.starts_with("_af_") {
                return Err(Error::Configuration(
                    "Redis storage names must be ASCII identifiers (up to 128 chars), excluding '_af_'."
                        .into(),
                ));
            }
            if field.is_full_text_indexed == Some(true) {
                return Err(Error::Configuration(
                    "The Redis vector store does not expose tokenized full-text or hybrid search."
                        .into(),
                ));
            }
            match field.field_type {
                FieldType::Vector => {
                    match field.index_kind.as_ref().map(|k| k.as_str()) {
                        None | Some("default" | "flat" | "hnsw") => {}
                        Some(other) => {
                            return Err(Error::Configuration(format!(
                                "Redis does not support index kind '{other}'."
                            )))
                        }
                    }
                    if redis_metric(field.distance_function.as_ref().map(|d| d.as_str())).is_none()
                    {
                        return Err(Error::Configuration(format!(
                            "Redis does not support distance function '{}'.",
                            field.distance_function.as_ref().map_or("", |d| d.as_str())
                        )));
                    }
                    match field.type_.as_deref() {
                        None | Some("float" | "float32" | "float64" | "bytes") => {}
                        Some(other) => {
                            return Err(Error::Configuration(format!(
                                "Redis does not support vector type '{other}'."
                            )))
                        }
                    }
                    if field.dimensions.is_none() {
                        return Err(Error::Configuration(
                            "Redis vector fields require dimensions.".into(),
                        ));
                    }
                }
                FieldType::Key | FieldType::Data => {
                    let kind = field.type_.as_deref();
                    if !matches!(
                        kind,
                        None | Some("str" | "bool" | "int" | "float" | "list" | "tuple" | "dict")
                    ) {
                        return Err(Error::Configuration(format!(
                            "Redis does not support data type '{}'.",
                            kind.unwrap_or_default()
                        )));
                    }
                    if field.field_type == FieldType::Key && !matches!(kind, None | Some("str")) {
                        return Err(Error::Configuration(
                            "Redis record keys must be strings.".into(),
                        ));
                    }
                    if Self::is_indexed(field) && kind == Some("dict") {
                        return Err(Error::Configuration(format!(
                            "Redis cannot index field '{}' of type 'dict'.",
                            field.name
                        )));
                    }
                }
            }
        }
        let (index_name, key_prefix) = collection_names(namespace, name)?;
        Ok(Self {
            conn,
            name: name.to_string(),
            definition,
            storage,
            index_name,
            key_prefix,
        })
    }

    fn is_indexed(field: &VectorStoreField) -> bool {
        field.field_type == FieldType::Key
            || (field.field_type == FieldType::Data && field.is_indexed == Some(true))
    }

    /// The Redis Search index name.
    pub fn index_name(&self) -> &str {
        &self.index_name
    }

    /// The prefix every document key in this collection carries.
    pub fn key_prefix(&self) -> &str {
        &self.key_prefix
    }

    /// The storage format.
    pub fn storage_type(&self) -> RedisStorageType {
        self.storage
    }

    fn indexed_fields(&self) -> impl Iterator<Item = &VectorStoreField> {
        self.definition
            .fields()
            .iter()
            .filter(|f| Self::is_indexed(f))
    }

    /// The `FT.CREATE` arguments after the command name.
    pub fn create_index_args(&self) -> Vec<String> {
        let mut args = vec![
            self.index_name.clone(),
            "ON".into(),
            self.storage.as_str().into(),
            "PREFIX".into(),
            "1".into(),
            self.key_prefix.clone(),
            "SCHEMA".into(),
        ];
        let json = self.storage == RedisStorageType::Json;
        let push_name = |args: &mut Vec<String>, name: &str| {
            if json {
                args.extend([format!("$.{name}"), "AS".into()]);
            }
            args.push(name.to_string());
        };
        for field in self.indexed_fields() {
            let name = field.effective_storage_name();
            push_name(&mut args, name);
            if matches!(kind_of(field), "int" | "float") {
                args.extend(["NUMERIC", "INDEXMISSING", "SORTABLE"].map(String::from));
            } else {
                args.extend(
                    [
                        "TAG".to_string(),
                        "SEPARATOR".into(),
                        TAG_SEPARATOR.to_string(),
                        "CASESENSITIVE".into(),
                        "INDEXEMPTY".into(),
                        "INDEXMISSING".into(),
                    ]
                    .into_iter()
                    .chain(
                        (!matches!(kind_of(field), "list" | "tuple")).then(|| "SORTABLE".into()),
                    ),
                );
            }
        }
        for field in self.definition.vector_fields() {
            let name = field.effective_storage_name();
            push_name(&mut args, name);
            let algorithm = match field.index_kind.as_ref().map(|k| k.as_str()) {
                Some("flat") => "FLAT",
                _ => "HNSW",
            };
            let metric = redis_metric(field.distance_function.as_ref().map(|d| d.as_str()))
                .expect("validated at construction");
            args.extend([
                "VECTOR".into(),
                algorithm.into(),
                "6".into(),
                "TYPE".into(),
                if is_float64(field) {
                    "FLOAT64"
                } else {
                    "FLOAT32"
                }
                .into(),
                "DIM".into(),
                field.dimensions.unwrap_or_default().to_string(),
                "DISTANCE_METRIC".into(),
                metric.into(),
            ]);
        }
        args
    }

    /// The Redis Search query for `options`' filters (`*` for none).
    pub fn prepare_filter(&self, options: &VectorSearchOptions) -> Result<String> {
        let mut parts = Vec::new();
        if let Some(filter) = &options.filter {
            parts.push(self.filter_expression(filter)?);
        }
        if let Some(raw) = options.provider_filter.as_deref().map(str::trim) {
            if !raw.is_empty() && raw != "*" {
                parts.push(format!("({raw})"));
            }
        }
        Ok(match parts.len() {
            0 => "*".into(),
            1 => parts.remove(0),
            _ => and(&parts),
        })
    }

    fn filter_expression(&self, expression: &FilterExpression) -> Result<String> {
        match expression {
            FilterExpression::Group(group) => {
                let parts = group
                    .filters
                    .iter()
                    .map(|c| self.filter_expression(c))
                    .collect::<Result<Vec<_>>>()?;
                match group.operator {
                    FilterGroupOperator::And => Ok(and(&parts)),
                    FilterGroupOperator::Or => or(&parts),
                    FilterGroupOperator::Not => {
                        // NOT negates the conjunction of its children.
                        Ok(not(&and(&parts)))
                    }
                }
            }
            FilterExpression::Condition(filter) => {
                let field = self
                    .definition
                    .try_get_field(&filter.field_name)
                    .filter(|f| Self::is_indexed(f))
                    .ok_or_else(|| {
                        Error::Configuration(format!(
                            "Redis filters require a declared, indexed, non-vector field; got '{}'.",
                            filter.field_name
                        ))
                    })?;
                filter_condition(filter, field, self.storage)
            }
        }
    }

    /// The `FT.SEARCH` query string for a KNN search.
    pub fn knn_query(
        &self,
        field: &VectorStoreField,
        options: &VectorSearchOptions,
    ) -> Result<String> {
        Ok(format!(
            "({})=>[KNN {} @{} $vector AS {DISTANCE_FIELD}]",
            self.prepare_filter(options)?,
            options.top + options.skip,
            field.effective_storage_name()
        ))
    }

    fn redis_key(&self, key: &Value) -> Result<String> {
        match key {
            Value::String(s) if !s.is_empty() => Ok(format!("{}{s}", self.key_prefix)),
            _ => Err(Error::Configuration(
                "Redis record keys must be nonempty strings.".into(),
            )),
        }
    }

    fn data_value(&self, value: &Value, field: &VectorStoreField) -> Result<Value> {
        if value.is_null() {
            if self.storage == RedisStorageType::Hash {
                return Err(Error::Configuration(
                    "Redis HASH has no native null value; use JSON storage for nullable fields."
                        .into(),
                ));
            }
            return Ok(Value::Null);
        }
        let kind = kind_of(field);
        let ok = match kind {
            "str" => value.is_string(),
            "bool" => value.is_boolean(),
            "int" => value.is_i64() || value.is_u64(),
            "float" => value.is_number(),
            "list" | "tuple" => value.is_array(),
            "dict" => value.is_object(),
            _ => false,
        };
        if !ok {
            return Err(Error::Configuration(format!(
                "Redis field '{}' requires {kind} values.",
                field.name
            )));
        }
        if Self::is_indexed(field) {
            match kind {
                "int" | "float" => {
                    numeric(value)?;
                }
                "str" => {
                    tag_value(value)?;
                }
                "list" | "tuple" => {
                    for item in value.as_array().into_iter().flatten() {
                        tag_value(item)?;
                    }
                }
                _ => {}
            }
        }
        json_finite(value)?;
        Ok(value.clone())
    }

    /// The HASH fields for a record (keyed by logical name), and its key.
    pub fn hash_document(&self, record: &Value) -> Result<(String, Vec<(String, HashValue)>)> {
        let stored = self.definition.to_storage(record)?;
        let key = self.record_key(&stored)?;
        let mut fields = Vec::new();
        for field in self.definition.fields() {
            let name = field.effective_storage_name();
            let Some(value) = stored.get(name) else {
                continue;
            };
            let encoded = if field.field_type == FieldType::Vector {
                if value.is_null() {
                    continue;
                }
                HashValue::Bytes(vector_bytes(&vector_values(value, field)?, field)?)
            } else {
                let value = self.data_value(value, field)?;
                let kind = kind_of(field);
                HashValue::Text(match (&value, kind) {
                    (Value::String(s), _) => s.clone(),
                    (Value::Bool(b), _) => b.to_string(),
                    (Value::Number(n), _) => n.to_string(),
                    (Value::Array(items), "list" | "tuple") if Self::is_indexed(field) => {
                        if items.is_empty() || items.iter().any(|i| i.as_str() == Some("")) {
                            return Err(Error::Configuration(
                                "Redis HASH TAG arrays cannot preserve empty arrays or empty elements."
                                    .into(),
                            ));
                        }
                        items
                            .iter()
                            .filter_map(Value::as_str)
                            .collect::<Vec<_>>()
                            .join(&TAG_SEPARATOR.to_string())
                    }
                    (other, _) => other.to_string(),
                })
            };
            fields.push((name.to_string(), encoded));
        }
        Ok((key, fields))
    }

    /// The JSON document for a record (keyed by logical name), and its key.
    pub fn json_document(&self, record: &Value) -> Result<(String, Value)> {
        let stored = self.definition.to_storage(record)?;
        let key = self.record_key(&stored)?;
        let mut document = Map::new();
        for field in self.definition.fields() {
            let name = field.effective_storage_name();
            let Some(value) = stored.get(name) else {
                continue;
            };
            let value = if field.field_type == FieldType::Vector {
                if value.is_null() {
                    Value::Null
                } else {
                    let values = vector_values(value, field)?;
                    vector_bytes(&values, field)?;
                    json!(values)
                }
            } else {
                self.data_value(value, field)?
            };
            document.insert(name.to_string(), value);
        }
        Ok((key, Value::Object(document)))
    }

    fn record_key(&self, stored: &Value) -> Result<String> {
        let key = stored
            .get(self.definition.key_field_storage_name())
            .ok_or_else(|| Error::Configuration("record is missing its key".into()))?;
        self.redis_key(key)
    }

    /// Decode the HASH values `HMGET` returned for `fields` into a record
    /// keyed by logical name.
    pub fn decode_hash(
        &self,
        fields: &[&VectorStoreField],
        values: &[Option<Vec<u8>>],
    ) -> Result<Value> {
        let mut record = Map::new();
        for (field, raw) in fields.iter().zip(values) {
            let Some(raw) = raw else {
                continue;
            };
            let value = if field.field_type == FieldType::Vector {
                decode_vector_bytes(raw, field)?
            } else {
                let text = std::str::from_utf8(raw).map_err(|_| {
                    Error::Service(format!("Redis field '{}' is not UTF-8.", field.name))
                })?;
                let invalid = || {
                    Error::Service(format!(
                        "Redis field '{}' does not hold a {} value.",
                        field.name,
                        kind_of(field)
                    ))
                };
                match kind_of(field) {
                    "str" => json!(text),
                    "int" => json!(text.parse::<i64>().map_err(|_| invalid())?),
                    "float" => json!(text.parse::<f64>().map_err(|_| invalid())?),
                    "bool" => match text {
                        "true" => json!(true),
                        "false" => json!(false),
                        _ => {
                            return Err(Error::Service(
                                "Redis boolean HASH fields must contain 'true' or 'false'.".into(),
                            ))
                        }
                    },
                    "list" | "tuple" if Self::is_indexed(field) => {
                        json!(text.split(TAG_SEPARATOR).collect::<Vec<_>>())
                    }
                    _ => serde_json::from_str(text).map_err(|_| invalid())?,
                }
            };
            record.insert(field.effective_storage_name().to_string(), value);
        }
        self.definition.from_storage(&Value::Object(record), true)
    }

    /// Decode a `JSON.GET key $.a $.b ...` reply into a record keyed by
    /// logical name.
    pub fn decode_json(&self, fields: &[&VectorStoreField], reply: &str) -> Result<Value> {
        let parsed: Value = serde_json::from_str(reply)
            .map_err(|e| Error::Service(format!("Redis returned invalid JSON: {e}")))?;
        // One path answers with the bare match array, several with an
        // object keyed by path.
        let lookup = |name: &str| -> Option<Value> {
            let matches = if fields.len() == 1 {
                parsed.as_array()
            } else {
                parsed.get(format!("$.{name}")).and_then(Value::as_array)
            };
            matches.and_then(|m| m.first().cloned())
        };
        let mut record = Map::new();
        for field in fields {
            let name = field.effective_storage_name();
            if let Some(value) = lookup(name) {
                record.insert(name.to_string(), value);
            }
        }
        self.definition.from_storage(&Value::Object(record), true)
    }

    async fn connection(&self) -> Result<redis::aio::MultiplexedConnection> {
        self.conn.get().await
    }

    async fn index_info(&self) -> Result<Option<RedisValue>> {
        let mut conn = self.connection().await?;
        match redis::cmd("FT.INFO")
            .arg(&self.index_name)
            .query_async::<RedisValue>(&mut conn)
            .await
        {
            Ok(info) => Ok(Some(info)),
            Err(e) if is_unknown_index(&e) => Ok(None),
            Err(e) => Err(map_redis_err(e)),
        }
    }

    async fn require_index(&self) -> Result<()> {
        match self.index_info().await? {
            None => Err(Error::Service(
                "Redis collection does not exist; call ensure_collection_exists() first.".into(),
            )),
            Some(info) => self.validate_schema(&info),
        }
    }

    /// Check an `FT.INFO` reply against this collection's definition.
    pub fn validate_schema(&self, info: &RedisValue) -> Result<()> {
        let incompatible = || {
            Error::Configuration(format!(
                "Redis index '{}' has an incompatible schema or document prefix.",
                self.index_name
            ))
        };
        let info = pairs(info);
        let definition = info
            .get("index_definition")
            .map(pairs)
            .ok_or_else(incompatible)?;
        let key_type = definition
            .get("key_type")
            .and_then(text)
            .ok_or_else(incompatible)?;
        let prefixes: Vec<String> = definition
            .get("prefixes")
            .map(list)
            .unwrap_or_default()
            .iter()
            .filter_map(text)
            .collect();
        if !key_type.eq_ignore_ascii_case(self.storage.as_str())
            || prefixes != [self.key_prefix.clone()]
        {
            return Err(incompatible());
        }
        let mut existing: Vec<(String, String, Option<String>, Option<String>)> = info
            .get("attributes")
            .map(list)
            .unwrap_or_default()
            .iter()
            .map(|attr| {
                // Attributes are flat [key, value, ...] lists with bare flags.
                let items = list(attr);
                let get = |k: &str| {
                    items
                        .windows(2)
                        .find(|w| text(&w[0]).is_some_and(|t| t.eq_ignore_ascii_case(k)))
                        .and_then(|w| text(&w[1]))
                };
                (
                    get("attribute").unwrap_or_default(),
                    get("type").unwrap_or_default().to_ascii_uppercase(),
                    get("dim"),
                    get("distance_metric").map(|m| m.to_ascii_uppercase()),
                )
            })
            .collect();
        let mut expected: Vec<(String, String, Option<String>, Option<String>)> = self
            .indexed_fields()
            .map(|f| {
                let kind = if matches!(kind_of(f), "int" | "float") {
                    "NUMERIC"
                } else {
                    "TAG"
                };
                (
                    f.effective_storage_name().to_string(),
                    kind.to_string(),
                    None,
                    None,
                )
            })
            .chain(self.definition.vector_fields().into_iter().map(|f| {
                (
                    f.effective_storage_name().to_string(),
                    "VECTOR".to_string(),
                    f.dimensions.map(|d| d.to_string()),
                    redis_metric(f.distance_function.as_ref().map(|d| d.as_str()))
                        .map(str::to_string),
                )
            }))
            .collect();
        // Older servers do not report vector parameters in FT.INFO; compare
        // them only where they are reported.
        for (e, x) in existing.iter_mut().zip(expected.iter_mut()) {
            if e.2.is_none() {
                x.2 = None;
            }
            if e.3.is_none() {
                x.3 = None;
            }
        }
        existing.sort();
        expected.sort();
        if existing != expected {
            return Err(incompatible());
        }
        Ok(())
    }

    async fn fetch(&self, keys: &[String], include_vectors: bool) -> Result<Vec<Option<Value>>> {
        let fields: Vec<&VectorStoreField> = self
            .definition
            .fields()
            .iter()
            .filter(|f| include_vectors || f.field_type != FieldType::Vector)
            .collect();
        let names: Vec<&str> = fields.iter().map(|f| f.effective_storage_name()).collect();
        let mut conn = self.connection().await?;
        let mut out = Vec::with_capacity(keys.len());
        for batch in keys.chunks(BATCH_SIZE) {
            let mut pipe = redis::pipe();
            for key in batch {
                if !key.starts_with(&self.key_prefix) {
                    return Err(Error::Service(
                        "Redis returned a key outside the collection prefix.".into(),
                    ));
                }
                match self.storage {
                    RedisStorageType::Hash => {
                        pipe.cmd("EXISTS").arg(key);
                        pipe.cmd("HMGET").arg(key).arg(&names);
                    }
                    RedisStorageType::Json => {
                        let paths: Vec<String> = names.iter().map(|n| format!("$.{n}")).collect();
                        pipe.cmd("JSON.GET").arg(key).arg(paths);
                    }
                }
            }
            let replies: Vec<RedisValue> =
                pipe.query_async(&mut conn).await.map_err(map_redis_err)?;
            match self.storage {
                RedisStorageType::Hash => {
                    for pair in replies.chunks(2) {
                        let exists = matches!(pair[0], RedisValue::Int(n) if n > 0);
                        if !exists {
                            out.push(None);
                            continue;
                        }
                        let values: Vec<Option<Vec<u8>>> = list(&pair[1])
                            .iter()
                            .map(|v| match v {
                                RedisValue::BulkString(b) => Some(b.clone()),
                                RedisValue::SimpleString(s) => Some(s.clone().into_bytes()),
                                _ => None,
                            })
                            .collect();
                        out.push(Some(self.decode_hash(&fields, &values)?));
                    }
                }
                RedisStorageType::Json => {
                    for reply in &replies {
                        match text(reply) {
                            None => out.push(None),
                            Some(body) => out.push(Some(self.decode_json(&fields, &body)?)),
                        }
                    }
                }
            }
        }
        Ok(out)
    }
}

fn is_unknown_index(e: &redis::RedisError) -> bool {
    let message = e.to_string().to_ascii_lowercase();
    message.contains("unknown index name") || message.contains("no such index")
}

/// A RESP value's text, when it is one.
fn text(value: &RedisValue) -> Option<String> {
    match value {
        RedisValue::BulkString(b) => String::from_utf8(b.clone()).ok(),
        RedisValue::SimpleString(s) => Some(s.clone()),
        RedisValue::VerbatimString { text, .. } => Some(text.clone()),
        RedisValue::Int(n) => Some(n.to_string()),
        RedisValue::Double(d) => Some(d.to_string()),
        RedisValue::Okay => Some("OK".into()),
        _ => None,
    }
}

fn list(value: &RedisValue) -> Vec<RedisValue> {
    match value {
        RedisValue::Array(items) | RedisValue::Set(items) => items.clone(),
        RedisValue::Map(pairs) => pairs
            .iter()
            .flat_map(|(k, v)| [k.clone(), v.clone()])
            .collect(),
        _ => Vec::new(),
    }
}

/// A RESP2 `[key, value, ...]` list or a RESP3 map, keyed by text.
fn pairs(value: &RedisValue) -> std::collections::HashMap<String, RedisValue> {
    list(value)
        .chunks(2)
        .filter(|c| c.len() == 2)
        .filter_map(|c| Some((text(&c[0])?, c[1].clone())))
        .collect()
}

#[async_trait]
impl VectorCollection for RedisVectorCollection {
    fn name(&self) -> &str {
        &self.name
    }

    fn definition(&self) -> &VectorStoreCollectionDefinition {
        &self.definition
    }

    async fn ensure_collection_exists(&self) -> Result<()> {
        let init_failed = |e: Error| {
            Error::Service(format!(
                "Redis vector index initialization failed ({e}). Requires Redis Search with \
                 INDEXMISSING/INDEXEMPTY and RedisJSON for JSON storage (tested minimum Redis 8.0.3)."
            ))
        };
        let mut conn = self.connection().await?;
        if self.storage == RedisStorageType::Json {
            // A read-only probe: PING does not establish RedisJSON support.
            redis::cmd("JSON.TYPE")
                .arg(format!("{}_af_capability_probe", self.key_prefix))
                .arg("$")
                .query_async::<RedisValue>(&mut conn)
                .await
                .map_err(|e| init_failed(map_redis_err(e)))?;
        }
        let info = match self.index_info().await.map_err(init_failed)? {
            Some(info) => info,
            None => {
                let created = redis::cmd("FT.CREATE")
                    .arg(self.create_index_args())
                    .query_async::<RedisValue>(&mut conn)
                    .await;
                match created {
                    Ok(_) => {}
                    // A concurrent creator won; validate what it made.
                    Err(e) if e.to_string().contains("Index already exists") => {}
                    Err(e) => return Err(init_failed(map_redis_err(e))),
                }
                self.index_info()
                    .await
                    .map_err(init_failed)?
                    .ok_or_else(|| init_failed(Error::Service("index not found".into())))?
            }
        };
        self.validate_schema(&info)
    }

    async fn collection_exists(&self) -> Result<bool> {
        Ok(self.index_info().await?.is_some())
    }

    async fn ensure_collection_deleted(&self) -> Result<()> {
        if let Some(info) = self.index_info().await? {
            self.validate_schema(&info)?;
            let mut conn = self.connection().await?;
            redis::cmd("FT.DROPINDEX")
                .arg(&self.index_name)
                .arg("DD")
                .query_async::<RedisValue>(&mut conn)
                .await
                .map_err(map_redis_err)?;
        }
        Ok(())
    }

    async fn upsert(&self, records: Vec<Value>) -> Result<Vec<Value>> {
        if records.is_empty() {
            return Ok(Vec::new());
        }
        let key_name = self.definition.key_field().name.clone();
        let keys = records
            .iter()
            .map(|r| {
                r.get(&key_name)
                    .cloned()
                    .ok_or_else(|| Error::Configuration("record is missing its key".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        enum Doc {
            Hash(String, Vec<(String, HashValue)>),
            Json(String, Value),
        }
        let docs = records
            .iter()
            .map(|r| match self.storage {
                RedisStorageType::Hash => self.hash_document(r).map(|(k, f)| Doc::Hash(k, f)),
                RedisStorageType::Json => self.json_document(r).map(|(k, d)| Doc::Json(k, d)),
            })
            .collect::<Result<Vec<_>>>()?;
        self.require_index().await?;
        let mut conn = self.connection().await?;
        for batch in docs.chunks(BATCH_SIZE) {
            let mut pipe = redis::pipe();
            pipe.atomic();
            for doc in batch {
                match doc {
                    Doc::Hash(key, fields) => {
                        pipe.cmd("DEL").arg(key).ignore();
                        if !fields.is_empty() {
                            let cmd = pipe.cmd("HSET").arg(key);
                            for (name, value) in fields {
                                match value {
                                    HashValue::Text(t) => cmd.arg(name).arg(t),
                                    HashValue::Bytes(b) => cmd.arg(name).arg(b.as_slice()),
                                };
                            }
                            cmd.ignore();
                        }
                    }
                    Doc::Json(key, document) => {
                        pipe.cmd("JSON.SET")
                            .arg(key)
                            .arg("$")
                            .arg(document.to_string())
                            .ignore();
                    }
                }
            }
            pipe.query_async::<()>(&mut conn)
                .await
                .map_err(map_redis_err)?;
        }
        Ok(keys)
    }

    async fn get(&self, keys: Vec<Value>, include_vectors: bool) -> Result<Vec<Option<Value>>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let redis_keys = keys
            .iter()
            .map(|k| self.redis_key(k))
            .collect::<Result<Vec<_>>>()?;
        self.require_index().await?;
        self.fetch(&redis_keys, include_vectors).await
    }

    async fn delete(&self, keys: Vec<Value>) -> Result<()> {
        let redis_keys = keys
            .iter()
            .map(|k| self.redis_key(k))
            .collect::<Result<Vec<_>>>()?;
        if redis_keys.is_empty() {
            return Ok(());
        }
        self.require_index().await?;
        let mut conn = self.connection().await?;
        for batch in redis_keys.chunks(BATCH_SIZE) {
            redis::cmd("DEL")
                .arg(batch)
                .query_async::<i64>(&mut conn)
                .await
                .map_err(map_redis_err)?;
        }
        Ok(())
    }

    async fn search(
        &self,
        vector: Vec<f32>,
        options: &VectorSearchOptions,
    ) -> Result<Vec<VectorSearchResult>> {
        options.validate()?;
        let field = self
            .definition
            .try_get_vector_field(options.vector_field_name.as_deref())
            .ok_or_else(|| {
                Error::Configuration("Redis search requires a declared vector field.".into())
            })?;
        let values: Vec<f64> = vector.iter().map(|v| *v as f64).collect();
        let blob = vector_bytes(&values, field)?;
        let query = self.knn_query(field, options)?;
        self.require_index().await?;
        let mut conn = self.connection().await?;
        let reply: RedisValue = redis::cmd("FT.SEARCH")
            .arg(&self.index_name)
            .arg(&query)
            .arg("RETURN")
            .arg(1)
            .arg(DISTANCE_FIELD)
            .arg("SORTBY")
            .arg(DISTANCE_FIELD)
            .arg("ASC")
            .arg("LIMIT")
            .arg(options.skip)
            .arg(options.top)
            .arg("PARAMS")
            .arg(2)
            .arg("vector")
            .arg(blob.as_slice())
            .arg("DIALECT")
            .arg(2)
            .query_async(&mut conn)
            .await
            .map_err(map_redis_err)?;
        let hits = parse_search_reply(&reply)?;
        let keys: Vec<String> = hits.iter().map(|(k, _)| k.clone()).collect();
        let records = self.fetch(&keys, options.include_vectors).await?;
        Ok(records
            .into_iter()
            .zip(hits)
            .filter_map(|(record, (_, score))| {
                record.map(|record| VectorSearchResult {
                    record,
                    score: Some(score),
                    score_kind: Some(VectorSearchResult::SCORE_KIND_DISTANCE.into()),
                })
            })
            .collect())
    }
}

/// The `(key, distance)` hits of an `FT.SEARCH ... RETURN 1 _af_distance`
/// reply.
pub fn parse_search_reply(reply: &RedisValue) -> Result<Vec<(String, f64)>> {
    let items = list(reply);
    let mut hits = Vec::new();
    let mut i = 1;
    while i < items.len() {
        let key = text(&items[i])
            .ok_or_else(|| Error::Service("Redis returned a non-text document key.".into()))?;
        let attributes = items.get(i + 1).map(pairs).unwrap_or_default();
        let score = attributes
            .get(DISTANCE_FIELD)
            .and_then(text)
            .and_then(|t| t.parse::<f64>().ok())
            .filter(|s| s.is_finite())
            .ok_or_else(|| Error::Service("Redis returned a non-finite vector distance.".into()))?;
        hits.push((key, score));
        i += 2;
    }
    Ok(hits)
}

/// A Redis server holding vector collections in one namespace. See the
/// [module docs](self).
#[derive(Clone)]
pub struct RedisVectorStore {
    conn: Arc<LazyConnection>,
    namespace: String,
    storage: RedisStorageType,
}

impl std::fmt::Debug for RedisVectorStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedisVectorStore")
            .field("namespace", &self.namespace)
            .field("storage", &self.storage)
            .finish_non_exhaustive()
    }
}

impl RedisVectorStore {
    /// A store for the server at `url`, in the `default` namespace with
    /// HASH storage. No connection is made until the first operation.
    pub fn new(url: &str) -> Result<Self> {
        let conn = LazyConnection::open(url)?;
        if conn.db() != 0 {
            return Err(Error::Configuration(
                "Redis vector stores require database 0; Redis Search cannot index other databases."
                    .into(),
            ));
        }
        Ok(Self {
            conn: Arc::new(conn),
            namespace: "default".into(),
            storage: RedisStorageType::Hash,
        })
    }

    /// From `REDIS_URL` (default `redis://localhost:6379`), as upstream.
    pub fn from_env() -> Result<Self> {
        Self::new(
            &std::env::var("REDIS_URL")
                .ok()
                .filter(|u| !u.is_empty())
                .unwrap_or_else(|| "redis://localhost:6379".into()),
        )
    }

    /// Isolate this store's indexes and documents under `namespace`.
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Result<Self> {
        let namespace = namespace.into();
        namespace_component(&namespace)?;
        self.namespace = namespace;
        Ok(self)
    }

    /// The storage format collections default to.
    pub fn with_storage_type(mut self, storage: RedisStorageType) -> Self {
        self.storage = storage;
        self
    }

    /// A typed collection handle in the store's storage format.
    pub fn collection(
        &self,
        name: &str,
        definition: VectorStoreCollectionDefinition,
    ) -> Result<RedisVectorCollection> {
        self.collection_with_storage(name, definition, self.storage)
    }

    /// A typed collection handle with an explicit storage format.
    pub fn collection_with_storage(
        &self,
        name: &str,
        definition: VectorStoreCollectionDefinition,
        storage: RedisStorageType,
    ) -> Result<RedisVectorCollection> {
        RedisVectorCollection::new(
            self.conn.clone(),
            &self.namespace,
            name,
            definition,
            storage,
        )
    }
}

#[async_trait]
impl VectorStore for RedisVectorStore {
    fn get_collection(
        &self,
        name: &str,
        definition: VectorStoreCollectionDefinition,
    ) -> Result<Box<dyn VectorCollection>> {
        Ok(Box::new(self.collection(name, definition)?))
    }

    async fn list_collection_names(&self) -> Result<Vec<String>> {
        let mut conn = self.conn.get().await?;
        let indexes: RedisValue = redis::cmd("FT._LIST")
            .query_async(&mut conn)
            .await
            .map_err(map_redis_err)?;
        let mut names: Vec<String> = list(&indexes)
            .iter()
            .filter_map(text)
            .filter_map(|index| parse_index_name(&self.namespace, &index))
            .collect();
        names.sort();
        Ok(names)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::vectors::{DistanceFunction, FilterGroup, IndexKind};

    fn definition() -> VectorStoreCollectionDefinition {
        VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id"),
            VectorStoreField::data("title").with_type("str").indexed(),
            VectorStoreField::data("year").with_type("int").indexed(),
            VectorStoreField::data("tags").with_type("list").indexed(),
            VectorStoreField::data("draft").with_type("bool").indexed(),
            VectorStoreField::data("meta").with_type("dict"),
            VectorStoreField::vector("embedding", 3).with_storage_name("vec"),
        ])
        .unwrap()
    }

    fn collection(storage: RedisStorageType) -> RedisVectorCollection {
        RedisVectorStore::new("redis://localhost:6379")
            .unwrap()
            .with_namespace("ns")
            .unwrap()
            .collection_with_storage("docs", definition(), storage)
            .unwrap()
    }

    fn where_(filter: impl Into<FilterExpression>) -> String {
        collection(RedisStorageType::Hash)
            .prepare_filter(&VectorSearchOptions::new(3).with_filter(filter))
            .unwrap()
    }

    #[test]
    fn names_are_hex_namespaced_and_round_trip() {
        let (index, prefix) = collection_names("ns", "docs").unwrap();
        assert_eq!(index, "af:vector:6e73:index:646f6373");
        assert_eq!(prefix, "af:vector:6e73:data:646f6373:");
        assert_eq!(parse_index_name("ns", &index).as_deref(), Some("docs"));
        assert_eq!(parse_index_name("other", &index), None);
        assert_eq!(parse_index_name("ns", "af:vector:6e73:index:ZZ"), None);
        assert_eq!(
            parse_index_name("ns", "af:vector:6e73:index:646F6373"),
            None
        );
        assert!(collection_names("", "docs").is_err());
    }

    #[test]
    fn a_non_zero_database_is_refused() {
        assert!(RedisVectorStore::new("redis://localhost:6379/2").is_err());
    }

    #[test]
    fn the_index_schema_matches_upstream() {
        let hash = collection(RedisStorageType::Hash).create_index_args();
        let joined = hash.join(" ");
        assert!(joined.starts_with(
            "af:vector:6e73:index:646f6373 ON HASH PREFIX 1 af:vector:6e73:data:646f6373: SCHEMA id TAG"
        ));
        assert!(joined.contains("year NUMERIC INDEXMISSING SORTABLE"));
        assert!(joined
            .contains("tags TAG SEPARATOR \u{1f} CASESENSITIVE INDEXEMPTY INDEXMISSING draft TAG"));
        assert!(joined
            .contains("title TAG SEPARATOR \u{1f} CASESENSITIVE INDEXEMPTY INDEXMISSING SORTABLE"));
        assert!(joined.ends_with("vec VECTOR HNSW 6 TYPE FLOAT32 DIM 3 DISTANCE_METRIC COSINE"));
        assert!(
            !joined.contains("meta"),
            "unindexed fields are not in the schema"
        );
        let json = collection(RedisStorageType::Json).create_index_args();
        assert!(json.join(" ").contains("ON JSON"));
        assert!(json.join(" ").contains("$.year AS year NUMERIC"));
    }

    #[test]
    fn invalid_definitions_are_refused() {
        let store = RedisVectorStore::new("redis://localhost").unwrap();
        let bad = |fields| {
            store
                .collection("c", VectorStoreCollectionDefinition::new(fields).unwrap())
                .is_err()
        };
        assert!(bad(vec![
            VectorStoreField::key("id"),
            VectorStoreField::data("_af_x"),
        ]));
        assert!(bad(vec![
            VectorStoreField::key("id"),
            VectorStoreField::data("t").full_text_indexed(),
        ]));
        assert!(bad(vec![
            VectorStoreField::key("id"),
            VectorStoreField::vector("v", 2)
                .with_distance_function(DistanceFunction::new(DistanceFunction::COSINE_SIMILARITY)),
        ]));
        assert!(bad(vec![
            VectorStoreField::key("id"),
            VectorStoreField::vector("v", 2).with_index_kind(IndexKind::new(IndexKind::DISK_ANN)),
        ]));
        assert!(bad(vec![VectorStoreField::key("id").with_type("int")]));
        assert!(bad(vec![
            VectorStoreField::key("id"),
            VectorStoreField::data("d").with_type("dict").indexed(),
        ]));
    }

    #[test]
    fn portable_filters_translate_to_redis_search() {
        assert_eq!(
            where_(Filter::eq("title", "a b").unwrap()),
            r"@title:{a\ b}"
        );
        assert_eq!(where_(Filter::eq("title", "").unwrap()), "@title:{\"\"}");
        assert_eq!(
            where_(Filter::eq("year", 2020).unwrap()),
            "@year:[2020 2020]"
        );
        assert_eq!(where_(Filter::eq("draft", true).unwrap()), "@draft:{true}");
        assert_eq!(
            where_(Filter::eq("year", "x").unwrap()),
            "(ismissing(@year) -ismissing(@year))"
        );
        assert_eq!(where_(Filter::gt("year", 5).unwrap()), "@year:[(5 +inf]");
        assert_eq!(where_(Filter::lte("year", 5).unwrap()), "@year:[-inf 5]");
        assert_eq!(
            where_(Filter::ne("title", "x").unwrap()),
            "((-(ismissing(@title))) (-(@title:{x})))"
        );
        assert_eq!(
            where_(Filter::any_of("title", vec![json!("a"), json!(null), json!("b")]).unwrap()),
            "(@title:{a} | @title:{b})"
        );
        assert_eq!(
            where_(Filter::contains_all("tags", vec![json!("x"), json!("y")]).unwrap()),
            "(@tags:{x} @tags:{y})"
        );
        let group = FilterGroup::or(vec![
            Filter::eq("year", 1).unwrap().into(),
            Filter::eq("year", 2).unwrap().into(),
        ])
        .unwrap();
        assert_eq!(where_(group), "(@year:[1 1] | @year:[2 2])");
    }

    #[test]
    fn unsupported_filters_are_refused() {
        let c = collection(RedisStorageType::Hash);
        let refused = |f: Filter| {
            c.prepare_filter(&VectorSearchOptions::new(3).with_filter(f))
                .is_err()
        };
        assert!(refused(Filter::eq("meta", "x").unwrap()), "unindexed");
        assert!(refused(Filter::gt("title", 1).unwrap()), "ordered on a tag");
        assert!(refused(Filter::eq("tags", "x").unwrap()), "list equality");
        assert!(
            refused(Filter::eq("year", 9_007_199_254_740_993_i64).unwrap()),
            "inexact"
        );
        assert!(refused(Filter::eq("title", " pad").unwrap()), "whitespace");
        let json = collection(RedisStorageType::Json);
        assert!(json
            .prepare_filter(
                &VectorSearchOptions::new(3).with_filter(Filter::is_not_null("title").unwrap())
            )
            .is_err());
    }

    #[test]
    fn a_provider_filter_is_conjoined() {
        let c = collection(RedisStorageType::Hash);
        let options = VectorSearchOptions::new(3)
            .with_filter(Filter::eq("year", 1).unwrap())
            .with_provider_filter("@title:{x}");
        assert_eq!(
            c.prepare_filter(&options).unwrap(),
            "(@year:[1 1] (@title:{x}))"
        );
        let field = c.definition().try_get_vector_field(None).unwrap().clone();
        assert_eq!(
            c.knn_query(&field, &VectorSearchOptions::new(3).with_skip(2))
                .unwrap(),
            "(*)=>[KNN 5 @vec $vector AS _af_distance]"
        );
    }

    #[test]
    fn hash_documents_round_trip() {
        let c = collection(RedisStorageType::Hash);
        let record = json!({
            "id": "a", "title": "T", "year": 2021, "tags": ["x", "y"], "draft": false,
            "meta": {"k": [1, 2]}, "embedding": [1.0, 0.5, -2.0],
        });
        let (key, fields) = c.hash_document(&record).unwrap();
        assert_eq!(key, "af:vector:6e73:data:646f6373:a");
        let get = |n: &str| fields.iter().find(|(k, _)| k == n).map(|(_, v)| v.clone());
        assert_eq!(get("draft"), Some(HashValue::Text("false".into())));
        assert_eq!(get("tags"), Some(HashValue::Text("x\u{1f}y".into())));
        assert_eq!(get("meta"), Some(HashValue::Text(r#"{"k":[1,2]}"#.into())));
        let Some(HashValue::Bytes(blob)) = get("vec") else {
            panic!("vector is binary")
        };
        assert_eq!(blob.len(), 12);

        let all: Vec<&VectorStoreField> = c.definition().fields().iter().collect();
        let raw: Vec<Option<Vec<u8>>> = all
            .iter()
            .map(|f| {
                get(f.effective_storage_name()).map(|v| match v {
                    HashValue::Text(t) => t.into_bytes(),
                    HashValue::Bytes(b) => b,
                })
            })
            .collect();
        assert_eq!(c.decode_hash(&all, &raw).unwrap(), record);
    }

    #[test]
    fn hash_storage_refuses_what_it_cannot_hold() {
        let c = collection(RedisStorageType::Hash);
        assert!(c.hash_document(&json!({"id": "a", "title": null})).is_err());
        assert!(c.hash_document(&json!({"id": "a", "tags": []})).is_err());
        assert!(c.hash_document(&json!({"id": "", "title": "x"})).is_err());
        assert!(c.hash_document(&json!({"id": "a", "year": "x"})).is_err());
        assert!(c
            .hash_document(&json!({"id": "a", "embedding": [0.0, 0.0, 0.0]}))
            .is_err());
        assert!(c
            .hash_document(&json!({"id": "a", "embedding": [1.0, 2.0]}))
            .is_err());
    }

    #[test]
    fn json_documents_round_trip_through_json_get() {
        let c = collection(RedisStorageType::Json);
        let record = json!({"id": "a", "title": null, "year": 1, "embedding": [1.0, 2.0, 3.0]});
        let (_, doc) = c.json_document(&record).unwrap();
        assert_eq!(doc["vec"], json!([1.0, 2.0, 3.0]));
        assert_eq!(doc["title"], Value::Null);
        let fields: Vec<&VectorStoreField> = c.definition().fields().iter().collect();
        let reply = json!({
            "$.id": ["a"], "$.title": [null], "$.year": [1], "$.tags": [], "$.draft": [],
            "$.meta": [], "$.vec": [[1.0, 2.0, 3.0]],
        })
        .to_string();
        assert_eq!(c.decode_json(&fields, &reply).unwrap(), record);
        let one = [fields[0]];
        assert_eq!(c.decode_json(&one, r#"["a"]"#).unwrap(), json!({"id": "a"}));
    }

    #[test]
    fn search_replies_parse_into_scored_keys() {
        let reply = RedisValue::Array(vec![
            RedisValue::Int(2),
            RedisValue::BulkString(b"k1".to_vec()),
            RedisValue::Array(vec![
                RedisValue::BulkString(b"_af_distance".to_vec()),
                RedisValue::BulkString(b"0.25".to_vec()),
            ]),
            RedisValue::BulkString(b"k2".to_vec()),
            RedisValue::Array(vec![
                RedisValue::BulkString(b"_af_distance".to_vec()),
                RedisValue::BulkString(b"0.5".to_vec()),
            ]),
        ]);
        assert_eq!(
            parse_search_reply(&reply).unwrap(),
            vec![("k1".into(), 0.25), ("k2".into(), 0.5)]
        );
        let bad = RedisValue::Array(vec![
            RedisValue::Int(1),
            RedisValue::BulkString(b"k".to_vec()),
            RedisValue::Array(vec![
                RedisValue::BulkString(b"_af_distance".to_vec()),
                RedisValue::BulkString(b"nan".to_vec()),
            ]),
        ]);
        assert!(parse_search_reply(&bad).is_err());
    }

    fn bulk(s: &str) -> RedisValue {
        RedisValue::BulkString(s.as_bytes().to_vec())
    }

    fn info_for(c: &RedisVectorCollection, prefix: &str) -> RedisValue {
        let attr = |name: &str, kind: &str, extra: &[&str]| {
            let mut items = vec![
                bulk("identifier"),
                bulk(name),
                bulk("attribute"),
                bulk(name),
                bulk("type"),
                bulk(kind),
            ];
            items.extend(extra.iter().map(|e| bulk(e)));
            RedisValue::Array(items)
        };
        RedisValue::Array(vec![
            bulk("index_name"),
            bulk(c.index_name()),
            bulk("index_definition"),
            RedisValue::Array(vec![
                bulk("key_type"),
                bulk("HASH"),
                bulk("prefixes"),
                RedisValue::Array(vec![bulk(prefix)]),
            ]),
            bulk("attributes"),
            RedisValue::Array(vec![
                attr("id", "TAG", &["SORTABLE"]),
                attr("title", "TAG", &[]),
                attr("year", "NUMERIC", &[]),
                attr("tags", "TAG", &[]),
                attr("draft", "TAG", &[]),
                attr(
                    "vec",
                    "VECTOR",
                    &["algorithm", "HNSW", "dim", "3", "distance_metric", "COSINE"],
                ),
            ]),
        ])
    }

    #[test]
    fn schema_validation_compares_prefix_and_attributes() {
        let c = collection(RedisStorageType::Hash);
        c.validate_schema(&info_for(&c, c.key_prefix())).unwrap();
        assert!(c.validate_schema(&info_for(&c, "other:")).is_err());
        let json = collection(RedisStorageType::Json);
        assert!(
            json.validate_schema(&info_for(&json, json.key_prefix()))
                .is_err(),
            "a HASH index does not serve a JSON collection"
        );
        let narrower = RedisVectorStore::new("redis://localhost")
            .unwrap()
            .with_namespace("ns")
            .unwrap()
            .collection(
                "docs",
                VectorStoreCollectionDefinition::new(vec![
                    VectorStoreField::key("id"),
                    VectorStoreField::vector("embedding", 4).with_storage_name("vec"),
                ])
                .unwrap(),
            )
            .unwrap();
        assert!(narrower
            .validate_schema(&info_for(&narrower, narrower.key_prefix()))
            .is_err());
    }

    #[test]
    fn tags_escape_every_non_word_character() {
        assert_eq!(escape_tag("a-b.c d_é"), r"a\-b\.c\ d_é");
    }
}
