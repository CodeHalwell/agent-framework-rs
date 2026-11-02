//! A Qdrant [`VectorStore`]: dense named-vector collections over the Qdrant
//! REST API.
//!
//! Port of upstream `agent_framework_qdrant` (`QdrantCollection`,
//! `QdrantStore`). Upstream drives the official Python client; this crate
//! speaks the REST API over `reqwest`, as the Azure AI Search store does.
//!
//! ```no_run
//! use agent_framework_core::vectors::{
//!     Filter, VectorSearchOptions, VectorStore, VectorStoreCollectionDefinition, VectorStoreField,
//! };
//! use agent_framework_qdrant::QdrantStore;
//!
//! # async fn demo() -> agent_framework_core::error::Result<()> {
//! let store = QdrantStore::new("http://localhost:6333");
//! let definition = VectorStoreCollectionDefinition::new(vec![
//!     VectorStoreField::key("id").with_type("int"),
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
//! - Every vector field is a **named** dense vector. The default metric is
//!   cosine similarity (higher is closer); `dot_prod`, `euclidean_distance`
//!   and `manhattan` are also supported. Index kinds: default/HNSW and flat
//!   (HNSW with `m = 0`).
//! - Keys are point ids: unsigned 64-bit integers or UUIDs (any string key
//!   must parse as a UUID; it is canonicalized to lowercase hyphenated
//!   form). Qdrant does not generate keys.
//! - Data fields are the point's payload. A field with
//!   [`indexed`](VectorStoreField::indexed) and a scalar type (`str`, `int`,
//!   `float`, `bool`) gets a payload index.
//! - Portable filters translate to Qdrant's `must` / `should` / `must_not`
//!   conditions. Text operators (`starts_with`, `ends_with`,
//!   `contains_text`) have no exact Qdrant equivalent and are refused.
//!   [`VectorSearchOptions::provider_filter`] is a Qdrant filter as JSON,
//!   conjoined with the portable one.
//!
//! # Divergences
//!
//! - Upstream's `qdrant.payload_index` field annotation has no counterpart:
//!   the Rust field definition has no provider annotations, so the index
//!   type follows the declared field type.
//! - Operation options (consistency, shard keys, timeouts) are not exposed.
//! - There is no local (in-process) mode; use the server.

use std::collections::HashMap;

use agent_framework_core::error::{Error, Result};
use agent_framework_core::vectors::{
    FieldType, Filter, FilterExpression, FilterGroupOperator, FilterOperator, VectorCollection,
    VectorSearchOptions, VectorSearchResult, VectorStore, VectorStoreCollectionDefinition,
    VectorStoreField,
};
use async_trait::async_trait;
use serde_json::{json, Map, Value};

/// Points per write or read request.
const BATCH_SIZE: usize = 256;
/// The largest integer a JSON number range operand holds exactly.
const MAX_EXACT_INTEGER: f64 = 9_007_199_254_740_991.0;

fn http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent(agent_framework_core::user_agent::user_agent(
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION"),
        ))
        .build()
        .unwrap_or_default()
}

/// The Qdrant metric name for a distance function (`None` is the default).
fn qdrant_distance(function: Option<&str>) -> Result<&'static str> {
    Ok(match function.unwrap_or("DEFAULT") {
        "DEFAULT" | "cosine_similarity" => "Cosine",
        "dot_prod" => "Dot",
        "euclidean_distance" => "Euclid",
        "manhattan" => "Manhattan",
        other => {
            return Err(Error::Configuration(format!(
                "Qdrant distance function '{other}' is not supported."
            )))
        }
    })
}

/// The payload index schema a data field needs, if any.
fn payload_index(field: &VectorStoreField) -> Result<Option<&'static str>> {
    if field.is_indexed != Some(true) {
        return Ok(None);
    }
    Ok(Some(match field.type_.as_deref() {
        Some("str") => "keyword",
        Some("int") => "integer",
        Some("float") => "float",
        Some("bool") => "bool",
        _ => {
            return Err(Error::Configuration(format!(
                "Indexed field '{}' requires a scalar type (str, int, float or bool).",
                field.name
            )))
        }
    }))
}

/// Validate and canonicalize a point id.
fn point_id(value: &Value) -> Result<Value> {
    match value {
        Value::Number(n) if n.is_u64() => Ok(value.clone()),
        Value::Number(_) => Err(Error::Configuration(
            "Qdrant integer keys must be unsigned 64-bit integers.".into(),
        )),
        Value::String(s) => uuid::Uuid::parse_str(s)
            .map(|u| Value::String(u.hyphenated().to_string()))
            .map_err(|_| {
                Error::Configuration(
                    "Qdrant string keys must be UUIDs; arbitrary strings are not supported.".into(),
                )
            }),
        _ => Err(Error::Configuration(
            "Qdrant keys must be unsigned 64-bit integers or UUIDs.".into(),
        )),
    }
}

/// The point id for a key used in a filter, or `None` when the value cannot
/// be one (so the condition matches nothing rather than failing).
fn filter_point_id(value: &Value, kind: Option<&str>) -> Option<Value> {
    match value {
        Value::Bool(_) => None,
        Value::Number(n) if matches!(kind, None | Some("int")) => {
            if let Some(u) = n.as_u64() {
                return Some(json!(u));
            }
            let f = n.as_f64()?;
            (f.is_finite() && f.fract() == 0.0 && f >= 0.0 && f <= u64::MAX as f64)
                .then(|| json!(f as u64))
        }
        Value::String(_) if matches!(kind, None | Some("str") | Some("UUID")) => {
            point_id(value).ok()
        }
        _ => None,
    }
}

fn dense_vector(value: &Value, name: &str) -> Result<Vec<f32>> {
    let Value::Array(items) = value else {
        return Err(Error::Configuration(format!(
            "Qdrant vector '{name}' must be a dense numeric array."
        )));
    };
    items
        .iter()
        .map(|item| {
            let n = item.as_f64().ok_or_else(|| {
                Error::Configuration(format!("Qdrant vector '{name}' must contain numbers."))
            })?;
            if !n.is_finite() || n.abs() > f32::MAX as f64 {
                return Err(Error::Configuration(format!(
                    "Qdrant vector '{name}' must contain finite float32 values."
                )));
            }
            Ok(n as f32)
        })
        .collect()
}

/// Check a payload value against the field's declared type.
fn field_payload(field: &VectorStoreField, value: &Value) -> Result<Value> {
    if let Value::Number(n) = value {
        if n.is_u64() && n.as_i64().is_none() {
            return Err(Error::Configuration(
                "Qdrant integer payloads must be signed 64-bit integers.".into(),
            ));
        }
    }
    let Some(kind) = field.type_.as_deref() else {
        return Ok(value.clone());
    };
    if value.is_null() {
        return Ok(Value::Null);
    }
    let ok = match kind {
        "str" => value.is_string(),
        "int" => value.is_i64(),
        "float" => value.is_number(),
        "bool" => value.is_boolean(),
        "list" | "tuple" | "set" | "Sequence" => value.is_array(),
        "dict" => value.is_object(),
        other => {
            return Err(Error::Configuration(format!(
                "Qdrant payload field type '{other}' is not supported."
            )))
        }
    };
    if !ok {
        return Err(Error::Configuration(format!(
            "Qdrant payload field '{}' must have declared type '{kind}'.",
            field.name
        )));
    }
    Ok(value.clone())
}

fn false_condition() -> Value {
    json!({"has_id": []})
}

fn presence(name: &str) -> Value {
    // values_count distinguishes a missing field from null and [].
    json!({"key": name, "values_count": {"gte": 0}})
}

fn is_null(name: &str) -> Value {
    json!({"is_null": {"key": name}})
}

fn non_null(name: &str) -> Value {
    json!({"must": [presence(name)], "must_not": [is_null(name)]})
}

fn range_operand(value: &Value) -> Result<f64> {
    let n = match value {
        Value::Number(n) => n.as_f64().unwrap_or(f64::NAN),
        _ => {
            return Err(Error::Configuration(
                "Qdrant ordered filters require numeric operands.".into(),
            ))
        }
    };
    if !n.is_finite() {
        return Err(Error::Configuration(
            "Qdrant numeric filter operands must be finite.".into(),
        ));
    }
    if n.abs() > MAX_EXACT_INTEGER {
        return Err(Error::Configuration(
            "Qdrant numeric range operands must be within +/- (2**53-1) to compare integers exactly."
                .into(),
        ));
    }
    Ok(n)
}

fn equality(name: &str, value: &Value, field: &VectorStoreField, element: bool) -> Result<Value> {
    if value.is_null() {
        if element {
            return Err(Error::Configuration(
                "Qdrant collection membership with a null operand is not supported.".into(),
            ));
        }
        return Ok(is_null(name));
    }
    if !matches!(value, Value::String(_) | Value::Bool(_) | Value::Number(_)) {
        return Err(Error::Configuration(
            "Qdrant equality and membership operands must be JSON scalars.".into(),
        ));
    }
    let kind = field.type_.as_deref();
    if !element {
        if !matches!(kind, Some("str" | "bool" | "int" | "float")) {
            return Err(Error::Configuration(format!(
                "Qdrant scalar equality requires a str, bool, int or float field type; '{}' has {kind:?}.",
                field.name
            )));
        }
        let mismatched = match value {
            Value::Bool(_) => kind != Some("bool"),
            Value::String(_) => kind != Some("str"),
            _ => !matches!(kind, Some("int" | "float")),
        };
        if mismatched {
            return Ok(false_condition());
        }
    }
    if matches!(value, Value::String(_) | Value::Bool(_)) {
        return Ok(json!({"key": name, "match": {"value": value}}));
    }
    if kind == Some("int") && !element {
        let Value::Number(n) = value else {
            unreachable!("checked above")
        };
        let integer = match n.as_i64() {
            Some(i) => Some(i),
            None => n
                .as_f64()
                .filter(|f| f.is_finite() && f.fract() == 0.0 && f.abs() < 9.2e18)
                .map(|f| f as i64),
        };
        return Ok(match integer {
            Some(i) => json!({"key": name, "match": {"value": i}}),
            None => false_condition(),
        });
    }
    let number = range_operand(value)?;
    // A range also matches an equal integer or float, and never a bool.
    Ok(json!({"key": name, "range": {"gte": number, "lte": number}}))
}

fn key_condition(filter: &Filter, field: &VectorStoreField) -> Result<Value> {
    let kind = field.type_.as_deref();
    let everything = || json!({"must_not": [false_condition()]});
    let value = filter.value.as_ref().unwrap_or(&Value::Null);
    let op = &filter.operator;
    Ok(match op {
        FilterOperator::Exists | FilterOperator::IsNotNull => everything(),
        FilterOperator::IsNull => false_condition(),
        FilterOperator::Eq if value.is_null() => false_condition(),
        FilterOperator::Ne if value.is_null() => everything(),
        FilterOperator::Eq | FilterOperator::Ne => {
            let ids: Vec<Value> = filter_point_id(value, kind).into_iter().collect();
            let has = json!({"has_id": ids});
            if *op == FilterOperator::Ne {
                json!({"must_not": [has]})
            } else {
                has
            }
        }
        FilterOperator::In | FilterOperator::NotIn => {
            let ids: Vec<Value> = value
                .as_array()
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|i| filter_point_id(i, kind))
                        .collect()
                })
                .unwrap_or_default();
            let has = json!({"has_id": ids});
            if *op == FilterOperator::NotIn {
                json!({"must_not": [has]})
            } else {
                has
            }
        }
        other => {
            return Err(Error::Configuration(format!(
                "Qdrant key filters do not support '{other:?}'."
            )))
        }
    })
}

/// A collection in a Qdrant server. See the [crate docs](crate).
#[derive(Clone)]
pub struct QdrantCollection {
    http: reqwest::Client,
    url: String,
    api_key: Option<String>,
    name: String,
    definition: VectorStoreCollectionDefinition,
}

impl std::fmt::Debug for QdrantCollection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QdrantCollection")
            .field("url", &self.url)
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl QdrantCollection {
    fn new(
        store: &QdrantStore,
        name: &str,
        definition: VectorStoreCollectionDefinition,
    ) -> Result<Self> {
        for field in definition.fields() {
            match field.field_type {
                FieldType::Key => match field.type_.as_deref() {
                    None | Some("int" | "str" | "UUID") => {}
                    Some(other) => {
                        return Err(Error::Configuration(format!(
                            "Qdrant keys must be int, str (UUID) or UUID, not '{other}'."
                        )))
                    }
                },
                FieldType::Vector => {
                    qdrant_distance(field.distance_function.as_ref().map(|d| d.as_str()))?;
                    match field.index_kind.as_ref().map(|k| k.as_str()) {
                        None | Some("default" | "hnsw" | "flat") => {}
                        Some(other) => {
                            return Err(Error::Configuration(format!(
                                "Qdrant index kind '{other}' is not supported."
                            )))
                        }
                    }
                }
                FieldType::Data => {
                    let name = field.effective_storage_name();
                    if name.is_empty() || name.contains(['.', '[', ']', '"', '\\']) {
                        return Err(Error::Configuration(
                            "Qdrant payload storage names cannot contain JSON-path punctuation."
                                .into(),
                        ));
                    }
                    if field.is_full_text_indexed == Some(true) {
                        return Err(Error::Configuration(
                            "Qdrant full-text indexes are not part of this dense-vector connector."
                                .into(),
                        ));
                    }
                    payload_index(field)?;
                }
            }
        }
        Ok(Self {
            http: store.http.clone(),
            url: store.url.clone(),
            api_key: store.api_key.clone(),
            name: name.to_string(),
            definition,
        })
    }

    fn collection_url(&self, suffix: &str) -> String {
        format!("{}/collections/{}{suffix}", self.url, self.name)
    }

    async fn call(
        &self,
        method: reqwest::Method,
        url: String,
        body: Option<Value>,
    ) -> Result<(u16, Value)> {
        call(&self.http, self.api_key.as_deref(), method, url, body).await
    }

    async fn ok(&self, method: reqwest::Method, url: String, body: Option<Value>) -> Result<Value> {
        let (status, value) = self.call(method, url, body).await?;
        expect_ok(status, value)
    }

    /// The Qdrant filter for `options`, or `None`.
    pub fn prepare_filter(&self, options: &VectorSearchOptions) -> Result<Option<Value>> {
        let mut must = Vec::new();
        if let Some(filter) = &options.filter {
            must.push(self.condition(filter)?);
        }
        if let Some(raw) = &options.provider_filter {
            must.push(serde_json::from_str(raw).map_err(|e| {
                Error::Configuration(format!("provider_filter is not a JSON Qdrant filter: {e}"))
            })?);
        }
        Ok((!must.is_empty()).then(|| json!({"must": must})))
    }

    fn condition(&self, expression: &FilterExpression) -> Result<Value> {
        let filter = match expression {
            FilterExpression::Group(group) => {
                let children = group
                    .filters
                    .iter()
                    .map(|c| self.condition(c))
                    .collect::<Result<Vec<_>>>()?;
                return Ok(match group.operator {
                    FilterGroupOperator::And => json!({"must": children}),
                    FilterGroupOperator::Or => json!({"should": children}),
                    // NOT negates the whole child, not each of its leaves.
                    FilterGroupOperator::Not => json!({"must_not": children}),
                });
            }
            FilterExpression::Condition(filter) => filter,
        };
        let field = self
            .definition
            .try_get_field(&filter.field_name)
            .filter(|_| !filter.field_name.contains('.'))
            .ok_or_else(|| {
                Error::Configuration(
                    "Qdrant filters support declared top-level logical fields only.".into(),
                )
            })?;
        match field.field_type {
            FieldType::Key => return key_condition(filter, field),
            FieldType::Vector => {
                return Err(Error::Configuration(
                    "Portable Qdrant filters do not operate on vector fields.".into(),
                ))
            }
            FieldType::Data => {}
        }
        let name = field.effective_storage_name();
        let value = filter.value.as_ref().unwrap_or(&Value::Null);
        let kind = field.type_.as_deref();
        Ok(match &filter.operator {
            FilterOperator::Exists => presence(name),
            FilterOperator::IsNull => is_null(name),
            FilterOperator::IsNotNull => non_null(name),
            FilterOperator::Eq => equality(name, value, field, false)?,
            FilterOperator::Ne => {
                json!({"must": [presence(name)], "must_not": [equality(name, value, field, false)?]})
            }
            op @ (FilterOperator::Gt
            | FilterOperator::Gte
            | FilterOperator::Lt
            | FilterOperator::Lte
            | FilterOperator::Between) => {
                if !matches!(kind, Some("int" | "float")) {
                    return Err(Error::Configuration(
                        "Qdrant ordered comparisons require a declared int or float field.".into(),
                    ));
                }
                let range = match op {
                    FilterOperator::Between => {
                        let bounds =
                            value.as_array().filter(|b| b.len() == 2).ok_or_else(|| {
                                Error::Configuration("between takes two bounds".into())
                            })?;
                        json!({"gte": range_operand(&bounds[0])?, "lte": range_operand(&bounds[1])?})
                    }
                    FilterOperator::Gt => json!({"gt": range_operand(value)?}),
                    FilterOperator::Gte => json!({"gte": range_operand(value)?}),
                    FilterOperator::Lt => json!({"lt": range_operand(value)?}),
                    _ => json!({"lte": range_operand(value)?}),
                };
                json!({"key": name, "range": range})
            }
            op @ (FilterOperator::In | FilterOperator::NotIn) => {
                let items = value.as_array().cloned().unwrap_or_default();
                let condition = if items.is_empty() {
                    false_condition()
                } else {
                    json!({"should": items
                        .iter()
                        .map(|i| equality(name, i, field, false))
                        .collect::<Result<Vec<_>>>()?})
                };
                if *op == FilterOperator::NotIn {
                    json!({"must": [non_null(name)], "must_not": [condition]})
                } else {
                    json!({"must": [non_null(name), condition]})
                }
            }
            op @ (FilterOperator::Contains
            | FilterOperator::ContainsAny
            | FilterOperator::ContainsAll) => {
                if !matches!(kind, Some("list" | "tuple" | "set" | "Sequence")) {
                    return Err(Error::Configuration(
                        "Qdrant collection membership requires a declared collection field type."
                            .into(),
                    ));
                }
                let operands = if *op == FilterOperator::Contains {
                    vec![value.clone()]
                } else {
                    value.as_array().cloned().unwrap_or_default()
                };
                let conditions = operands
                    .iter()
                    .map(|i| equality(name, i, field, true))
                    .collect::<Result<Vec<_>>>()?;
                if *op == FilterOperator::ContainsAll {
                    let mut must = vec![non_null(name)];
                    must.extend(conditions);
                    json!({"must": must})
                } else if conditions.is_empty() {
                    false_condition()
                } else {
                    json!({"should": conditions})
                }
            }
            other => {
                return Err(Error::Configuration(format!(
                    "Qdrant does not support portable filter operator '{other:?}'. Literal text \
                     operations are not equivalent to tokenized full-text search."
                )))
            }
        })
    }

    /// The point for a record.
    fn point(&self, record: &Value) -> Result<Value> {
        let key_field = self.definition.key_field();
        let raw_key = record
            .get(key_field.effective_storage_name())
            .ok_or_else(|| Error::Configuration("record has no key".into()))?;
        let id = point_id(raw_key)?;
        match key_field.type_.as_deref() {
            Some("int") if !id.is_u64() => {
                return Err(Error::Configuration(
                    "Qdrant key must match declared type 'int'.".into(),
                ))
            }
            Some("str" | "UUID") if !id.is_string() => {
                return Err(Error::Configuration(
                    "Qdrant key must match the declared string/UUID type.".into(),
                ))
            }
            _ => {}
        }
        let mut payload = Map::new();
        for field in self.definition.data_fields() {
            let name = field.effective_storage_name();
            let value = record.get(name).unwrap_or(&Value::Null);
            payload.insert(name.to_string(), field_payload(field, value)?);
        }
        let mut vectors = Map::new();
        for field in self.definition.vector_fields() {
            let name = field.effective_storage_name();
            match record.get(name) {
                None | Some(Value::Null) => {}
                Some(v) => {
                    vectors.insert(name.to_string(), json!(dense_vector(v, &field.name)?));
                }
            }
        }
        Ok(json!({"id": id, "payload": payload, "vector": vectors}))
    }

    /// The record for a returned point.
    fn record(&self, point: &Value) -> Result<Value> {
        let mut record = match point.get("payload") {
            Some(Value::Object(p)) => p.clone(),
            _ => Map::new(),
        };
        let id = point.get("id").cloned().unwrap_or(Value::Null);
        record.insert(self.definition.key_field_storage_name().to_string(), id);
        match point.get("vector") {
            None | Some(Value::Null) => {}
            Some(Value::Object(vectors)) => {
                for field in self.definition.vector_fields() {
                    let name = field.effective_storage_name();
                    record.insert(
                        name.to_string(),
                        vectors.get(name).cloned().unwrap_or(Value::Null),
                    );
                }
            }
            Some(_) => {
                return Err(Error::Service(
                    "Expected named vectors in the Qdrant response.".into(),
                ))
            }
        }
        Ok(Value::Object(record))
    }
}

async fn call(
    http: &reqwest::Client,
    api_key: Option<&str>,
    method: reqwest::Method,
    url: String,
    body: Option<Value>,
) -> Result<(u16, Value)> {
    let mut request = http.request(method, &url);
    if let Some(key) = api_key {
        request = request.header("api-key", key);
    }
    if let Some(body) = body {
        request = request.json(&body);
    }
    let response = request.send().await.map_err(|e| {
        Error::Service(format!(
            "Qdrant request failed: {}",
            agent_framework_core::error::error_chain(&e)
        ))
    })?;
    let status = response.status().as_u16();
    let text = response
        .text()
        .await
        .map_err(|e| Error::Service(format!("Qdrant response could not be read: {e}")))?;
    let value = if text.trim().is_empty() {
        Value::Null
    } else {
        serde_json::from_str(&text).unwrap_or(Value::String(text))
    };
    Ok((status, value))
}

/// The `result` of a 2xx answer, or the error it carries.
fn expect_ok(status: u16, value: Value) -> Result<Value> {
    if (200..300).contains(&status) {
        return Ok(value.get("result").cloned().unwrap_or(Value::Null));
    }
    let detail = value
        .pointer("/status/error")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| value.to_string());
    Err(Error::Service(format!(
        "Qdrant returned {status}: {detail}"
    )))
}

fn expect_completed(result: &Value) -> Result<()> {
    match result.get("status").and_then(Value::as_str) {
        Some("completed") => Ok(()),
        other => Err(Error::Service(format!(
            "Qdrant write did not complete: {other:?}."
        ))),
    }
}

#[async_trait]
impl VectorCollection for QdrantCollection {
    fn name(&self) -> &str {
        &self.name
    }

    fn definition(&self) -> &VectorStoreCollectionDefinition {
        &self.definition
    }

    async fn ensure_collection_exists(&self) -> Result<()> {
        if !self.collection_exists().await? {
            let mut vectors = Map::new();
            for field in self.definition.vector_fields() {
                let mut params = json!({
                    "size": field.dimensions.unwrap_or_default(),
                    "distance": qdrant_distance(field.distance_function.as_ref().map(|d| d.as_str()))?,
                });
                if field.index_kind.as_ref().map(|k| k.as_str()) == Some("flat") {
                    params["hnsw_config"] = json!({"m": 0});
                }
                vectors.insert(field.effective_storage_name().to_string(), params);
            }
            let (status, value) = self
                .call(
                    reqwest::Method::PUT,
                    self.collection_url(""),
                    Some(json!({"vectors": vectors})),
                )
                .await?;
            // 409: a concurrent creator won; validate what it made below.
            if status != 409 {
                expect_ok(status, value)?;
            }
        }
        // A creator that just won may not have initialized its shards yet.
        let mut attempt = 0;
        let info = loop {
            let (status, value) = self
                .call(reqwest::Method::GET, self.collection_url(""), None)
                .await?;
            let not_ready =
                status == 500 && value.to_string().contains("0 of 0 read operations failed");
            if not_ready && attempt < 5 {
                tokio::time::sleep(std::time::Duration::from_millis(50 << attempt)).await;
                attempt += 1;
                continue;
            }
            break expect_ok(status, value)?;
        };
        let configured = info
            .pointer("/config/params/vectors")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                Error::Configuration(
                    "QdrantCollection requires named vectors, not an unnamed-vector collection."
                        .into(),
                )
            })?;
        let collection_m = info
            .pointer("/config/hnsw_config/m")
            .and_then(Value::as_u64);
        for field in self.definition.vector_fields() {
            let vector = configured.get(field.effective_storage_name());
            let distance = qdrant_distance(field.distance_function.as_ref().map(|d| d.as_str()))?;
            let matches = vector.is_some_and(|v| {
                v.get("size").and_then(Value::as_u64) == field.dimensions.map(|d| d as u64)
                    && v.get("distance").and_then(Value::as_str) == Some(distance)
                    && v.get("multivector_config").is_none_or(Value::is_null)
                    && v.get("datatype")
                        .and_then(Value::as_str)
                        .is_none_or(|d| d.eq_ignore_ascii_case("float32"))
            });
            if !matches {
                return Err(Error::Configuration(format!(
                    "Existing Qdrant vector '{}' does not match the collection definition.",
                    field.name
                )));
            }
            let m = vector
                .and_then(|v| v.pointer("/hnsw_config/m"))
                .and_then(Value::as_u64)
                .or(collection_m);
            match field.index_kind.as_ref().map(|k| k.as_str()) {
                Some("flat") if m != Some(0) => {
                    return Err(Error::Configuration(format!(
                        "Existing Qdrant vector '{}' does not use a flat index.",
                        field.name
                    )))
                }
                Some("hnsw") if m == Some(0) => {
                    return Err(Error::Configuration(format!(
                        "Existing Qdrant vector '{}' does not use an HNSW index.",
                        field.name
                    )))
                }
                _ => {}
            }
        }
        let schema = info.get("payload_schema").cloned().unwrap_or(Value::Null);
        for field in self.definition.data_fields() {
            let Some(index) = payload_index(field)? else {
                continue;
            };
            let name = field.effective_storage_name();
            match schema
                .get(name)
                .and_then(|s| s.get("data_type"))
                .and_then(Value::as_str)
            {
                Some(existing) if existing != index => {
                    return Err(Error::Configuration(format!(
                        "Existing Qdrant payload index '{name}' has a different type."
                    )))
                }
                Some(_) => {}
                None => {
                    let result = self
                        .ok(
                            reqwest::Method::PUT,
                            self.collection_url("/index?wait=true"),
                            Some(json!({"field_name": name, "field_schema": index})),
                        )
                        .await?;
                    expect_completed(&result)?;
                }
            }
        }
        Ok(())
    }

    async fn collection_exists(&self) -> Result<bool> {
        let result = self
            .ok(reqwest::Method::GET, self.collection_url("/exists"), None)
            .await?;
        Ok(result
            .get("exists")
            .and_then(Value::as_bool)
            .unwrap_or(false))
    }

    async fn ensure_collection_deleted(&self) -> Result<()> {
        if self.collection_exists().await? {
            let deleted = self
                .ok(reqwest::Method::DELETE, self.collection_url(""), None)
                .await?;
            if deleted != Value::Bool(true) {
                return Err(Error::Service(format!(
                    "Qdrant did not delete collection '{}'.",
                    self.name
                )));
            }
        }
        Ok(())
    }

    async fn upsert(&self, records: Vec<Value>) -> Result<Vec<Value>> {
        let points = records
            .iter()
            .map(|r| self.point(r))
            .collect::<Result<Vec<_>>>()?;
        for batch in points.chunks(BATCH_SIZE) {
            let result = self
                .ok(
                    reqwest::Method::PUT,
                    self.collection_url("/points?wait=true"),
                    Some(json!({"points": batch})),
                )
                .await?;
            expect_completed(&result)?;
        }
        Ok(points.into_iter().map(|p| p["id"].clone()).collect())
    }

    async fn get(&self, keys: Vec<Value>, include_vectors: bool) -> Result<Vec<Option<Value>>> {
        let ids = keys.iter().map(point_id).collect::<Result<Vec<_>>>()?;
        let mut found: HashMap<String, Value> = HashMap::new();
        for batch in ids.chunks(BATCH_SIZE) {
            let result = self
                .ok(
                    reqwest::Method::POST,
                    self.collection_url("/points"),
                    Some(
                        json!({"ids": batch, "with_payload": true, "with_vector": include_vectors}),
                    ),
                )
                .await?;
            for point in result.as_array().cloned().unwrap_or_default() {
                let id = point.get("id").cloned().unwrap_or(Value::Null);
                found.insert(id.to_string(), self.record(&point)?);
            }
        }
        Ok(ids
            .iter()
            .map(|id| found.get(&id.to_string()).cloned())
            .collect())
    }

    async fn delete(&self, keys: Vec<Value>) -> Result<()> {
        let ids = keys.iter().map(point_id).collect::<Result<Vec<_>>>()?;
        for batch in ids.chunks(BATCH_SIZE) {
            let result = self
                .ok(
                    reqwest::Method::POST,
                    self.collection_url("/points/delete?wait=true"),
                    Some(json!({"points": batch})),
                )
                .await?;
            expect_completed(&result)?;
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
                Error::Configuration("Qdrant search requires a declared vector field.".into())
            })?;
        if let Some(dims) = field.dimensions {
            if vector.len() != dims {
                return Err(Error::Configuration(format!(
                    "query vector has {} dimensions; '{}' has {dims}",
                    vector.len(),
                    field.name
                )));
            }
        }
        if vector.iter().any(|v| !v.is_finite()) {
            return Err(Error::Configuration(
                "Qdrant query vectors must contain finite values.".into(),
            ));
        }
        let mut body = json!({
            "query": vector,
            "using": field.effective_storage_name(),
            "limit": options.top,
            "offset": options.skip,
            "with_payload": true,
            "with_vector": options.include_vectors,
        });
        if let Some(filter) = self.prepare_filter(options)? {
            body["filter"] = filter;
        }
        let result = self
            .ok(
                reqwest::Method::POST,
                self.collection_url("/points/query"),
                Some(body),
            )
            .await?;
        result
            .get("points")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
            .iter()
            .map(|point| {
                Ok(VectorSearchResult {
                    record: self.record(point)?,
                    score: point.get("score").and_then(Value::as_f64),
                    score_kind: None,
                })
            })
            .collect()
    }
}

/// A connection to a Qdrant server. See the [crate docs](crate).
#[derive(Clone)]
pub struct QdrantStore {
    http: reqwest::Client,
    url: String,
    api_key: Option<String>,
}

impl std::fmt::Debug for QdrantStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QdrantStore")
            .field("url", &self.url)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

impl QdrantStore {
    /// A store for the server at `url` (for example `http://localhost:6333`).
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            http: http_client(),
            url: url.into().trim_end_matches('/').to_string(),
            api_key: None,
        }
    }

    /// From `QDRANT_URL` (default `http://localhost:6333`) and
    /// `QDRANT_API_KEY`, as upstream's settings.
    pub fn from_env() -> Self {
        let store = Self::new(
            std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6333".into()),
        );
        match std::env::var("QDRANT_API_KEY") {
            Ok(key) if !key.is_empty() => store.with_api_key(key),
            _ => store,
        }
    }

    /// Authenticate with an API key (sent as the `api-key` header).
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Use this HTTP client.
    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    /// A typed collection handle.
    pub fn collection(
        &self,
        name: &str,
        definition: VectorStoreCollectionDefinition,
    ) -> Result<QdrantCollection> {
        QdrantCollection::new(self, name, definition)
    }
}

#[async_trait]
impl VectorStore for QdrantStore {
    fn get_collection(
        &self,
        name: &str,
        definition: VectorStoreCollectionDefinition,
    ) -> Result<Box<dyn VectorCollection>> {
        Ok(Box::new(self.collection(name, definition)?))
    }

    async fn list_collection_names(&self) -> Result<Vec<String>> {
        let (status, value) = call(
            &self.http,
            self.api_key.as_deref(),
            reqwest::Method::GET,
            format!("{}/collections", self.url),
            None,
        )
        .await?;
        let result = expect_ok(status, value)?;
        Ok(result
            .get("collections")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|c| c.get("name").and_then(Value::as_str).map(str::to_string))
                    .collect()
            })
            .unwrap_or_default())
    }

    async fn collection_exists(&self, name: &str) -> Result<bool> {
        let (status, value) = call(
            &self.http,
            self.api_key.as_deref(),
            reqwest::Method::GET,
            format!("{}/collections/{name}/exists", self.url),
            None,
        )
        .await?;
        Ok(expect_ok(status, value)?
            .get("exists")
            .and_then(Value::as_bool)
            .unwrap_or(false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::vectors::{DistanceFunction, FilterGroup};

    fn collection() -> QdrantCollection {
        let definition = VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id").with_type("int"),
            VectorStoreField::data("year").with_type("int").indexed(),
            VectorStoreField::data("genre").with_type("str"),
            VectorStoreField::data("tags").with_type("list"),
            VectorStoreField::vector("embedding", 2),
        ])
        .unwrap();
        QdrantStore::new("http://localhost:6333")
            .collection("docs", definition)
            .unwrap()
    }

    fn filter_of(expression: FilterExpression) -> Value {
        collection()
            .prepare_filter(&VectorSearchOptions::new(1).with_filter(expression))
            .unwrap()
            .unwrap()
    }

    #[test]
    fn point_ids_are_u64_or_canonical_uuids() {
        assert_eq!(point_id(&json!(7)).unwrap(), json!(7));
        assert!(point_id(&json!(-1)).is_err());
        assert_eq!(
            point_id(&json!("A1B2C3D4E5F60718293A4B5C6D7E8F90")).unwrap(),
            json!("a1b2c3d4-e5f6-0718-293a-4b5c6d7e8f90")
        );
        assert!(point_id(&json!("not-a-uuid")).is_err());
        assert!(point_id(&json!(true)).is_err());
    }

    #[test]
    fn distances_and_unsupported_definitions() {
        assert_eq!(qdrant_distance(None).unwrap(), "Cosine");
        assert_eq!(
            qdrant_distance(Some("euclidean_distance")).unwrap(),
            "Euclid"
        );
        assert!(qdrant_distance(Some("hamming")).is_err());
        let bad = VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id"),
            VectorStoreField::vector("v", 2)
                .with_distance_function(DistanceFunction::new("cosine_distance")),
        ])
        .unwrap();
        assert!(QdrantStore::new("http://x").collection("c", bad).is_err());
        let indexed_untyped = VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id"),
            VectorStoreField::data("x").indexed(),
            VectorStoreField::vector("v", 2),
        ])
        .unwrap();
        assert!(QdrantStore::new("http://x")
            .collection("c", indexed_untyped)
            .is_err());
    }

    #[test]
    fn portable_filters_translate() {
        assert_eq!(
            filter_of(Filter::gte("year", 2020).unwrap().into()),
            json!({"must": [{"key": "year", "range": {"gte": 2020.0}}]})
        );
        assert_eq!(
            filter_of(Filter::eq("genre", "sf").unwrap().into()),
            json!({"must": [{"key": "genre", "match": {"value": "sf"}}]})
        );
        // A type-mismatched equality matches nothing rather than failing.
        assert_eq!(
            filter_of(Filter::eq("genre", 3).unwrap().into()),
            json!({"must": [{"has_id": []}]})
        );
        assert_eq!(
            filter_of(Filter::eq("id", 5).unwrap().into()),
            json!({"must": [{"has_id": [5]}]})
        );
        let group = FilterGroup::or(vec![
            Filter::eq("year", 1999).unwrap().into(),
            FilterGroup::not(Filter::is_null("genre").unwrap().into()).unwrap(),
        ])
        .unwrap();
        assert_eq!(
            filter_of(group),
            json!({"must": [{"should": [
                {"key": "year", "match": {"value": 1999}},
                {"must_not": [{"is_null": {"key": "genre"}}]}
            ]}]})
        );
        let starts = Filter::new("genre", FilterOperator::StartsWith, Some(json!("s"))).unwrap();
        assert!(collection()
            .prepare_filter(&VectorSearchOptions::new(1).with_filter(starts))
            .is_err());
    }

    #[test]
    fn points_round_trip_through_records() {
        let c = collection();
        let point = c
            .point(&json!({"id": 1, "year": 2001, "genre": "sf", "tags": ["a"], "embedding": [0.5, 1.0]}))
            .unwrap();
        assert_eq!(
            point["payload"],
            json!({"year": 2001, "genre": "sf", "tags": ["a"]})
        );
        assert_eq!(point["vector"], json!({"embedding": [0.5, 1.0]}));
        assert!(c
            .point(
                &json!({"id": 1, "year": "x", "genre": "sf", "tags": [], "embedding": [0.0, 0.0]})
            )
            .is_err());
        let record = c.record(&point).unwrap();
        assert_eq!(record["id"], json!(1));
        assert_eq!(record["embedding"], json!([0.5, 1.0]));
    }
}
