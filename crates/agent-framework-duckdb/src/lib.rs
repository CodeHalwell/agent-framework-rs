//! A DuckDB [`VectorStore`]: one table per collection, typed columns, and
//! exact dense-vector search with DuckDB's array distance functions.
//!
//! Port of upstream `agent_framework_duckdb` (`DuckDBCollection`,
//! `DuckDBStore`) over the `duckdb` crate, bundled (DuckDB is compiled in,
//! so there is nothing to install).
//!
//! ```no_run
//! use agent_framework_core::vectors::{
//!     Filter, VectorSearchOptions, VectorStore, VectorStoreCollectionDefinition, VectorStoreField,
//! };
//! use agent_framework_duckdb::DuckDbStore;
//!
//! # async fn demo() -> agent_framework_core::error::Result<()> {
//! let store = DuckDbStore::open("notes.duckdb")?;
//! let definition = VectorStoreCollectionDefinition::new(vec![
//!     VectorStoreField::key("id").with_type("str"),
//!     VectorStoreField::data("year").with_type("int"),
//!     VectorStoreField::vector("embedding", 3),
//! ])?;
//! let notes = store.get_collection("notes", definition)?;
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
//! - Every field needs a declared type: `str`, `int`, `float`, `bool`,
//!   `UUID`, `date`, `datetime` (timezone required), `list` and `dict`
//!   (stored as JSON). Keys are `str`, `int` or `UUID`.
//! - Vectors are fixed-size arrays (`float32` → `FLOAT[n]`, `float` /
//!   `float64` → `DOUBLE[n]`). Search is exact. Distance functions:
//!   `DEFAULT` / `cosine_distance` (lower is closer), `cosine_similarity`
//!   and `dot_prod` (higher is closer), `euclidean_distance`, and
//!   `negative_dot_prod`.
//! - Portable filters compile to parameterized SQL; text operators use
//!   `LIKE` with escaping. JSON fields are not filterable.
//! - Statements run one at a time on a single connection, on Tokio's
//!   blocking pool.
//!
//! # Divergences
//!
//! - The `bytes` type is refused: records are JSON, which has no bytes.
//! - There is no `score_threshold` (the Rust search options carry none).

#![cfg(feature = "experimental-vector-stores")]
#![cfg_attr(docsrs, doc(cfg(feature = "experimental-vector-stores")))]

use std::sync::{Arc, Mutex};

use agent_framework_core::error::{Error, Result};
use agent_framework_core::vectors::{
    FieldType, Filter, FilterExpression, FilterGroupOperator, FilterOperator, VectorCollection,
    VectorSearchOptions, VectorSearchResult, VectorStore, VectorStoreCollectionDefinition,
    VectorStoreField,
};
use async_trait::async_trait;
use duckdb::types::Value as Db;
use serde_json::Value;

/// The database file used when none is given, as upstream.
pub const DEFAULT_DATABASE: &str = "agent-framework.duckdb";

fn map_err(e: duckdb::Error) -> Error {
    Error::service(format!("DuckDB operation failed: {e}"))
}

/// Quote an identifier.
fn ident(name: &str) -> Result<String> {
    if name.is_empty() || name.contains('\0') {
        return Err(Error::Configuration(
            "DuckDB identifiers must be nonempty strings without NUL.".into(),
        ));
    }
    Ok(format!("\"{}\"", name.replace('"', "\"\"")))
}

/// A single-quoted SQL string literal.
fn literal(text: &str) -> String {
    format!("'{}'", text.replace('\'', "''"))
}

fn column_type(field: &VectorStoreField) -> Result<String> {
    if field.field_type == FieldType::Vector {
        let dims = field.dimensions.filter(|d| *d > 0).ok_or_else(|| {
            Error::Configuration(format!(
                "Vector field '{}' needs positive dimensions.",
                field.name
            ))
        })?;
        let element = match field.type_.as_deref() {
            None | Some("float") | Some("float64") => "DOUBLE",
            Some("float32") => "FLOAT",
            Some(other) => {
                return Err(Error::Configuration(format!(
                    "Unsupported DuckDB vector type '{other}'."
                )))
            }
        };
        return Ok(format!("{element}[{dims}]"));
    }
    Ok(match field.type_.as_deref() {
        Some("str") => "VARCHAR",
        Some("int") => "BIGINT",
        Some("float") => "DOUBLE",
        Some("bool") => "BOOLEAN",
        Some("UUID") => "UUID",
        Some("date") => "DATE",
        Some("datetime") => "TIMESTAMPTZ",
        Some("list") | Some("dict") => "JSON",
        other => {
            return Err(Error::Configuration(format!(
                "Field '{}' needs a supported explicit type; got {other:?}.",
                field.name
            )))
        }
    }
    .to_string())
}

/// `(function, higher_is_closer)` for a distance function.
fn metric(field: &VectorStoreField) -> Result<(&'static str, bool)> {
    Ok(
        match field
            .distance_function
            .as_ref()
            .map(|d| d.as_str())
            .unwrap_or("DEFAULT")
        {
            "DEFAULT" | "cosine_distance" => ("array_cosine_distance", false),
            "cosine_similarity" => ("array_cosine_similarity", true),
            "euclidean_distance" => ("array_distance", false),
            "dot_prod" => ("array_inner_product", true),
            "negative_dot_prod" => ("-array_inner_product", false),
            other => {
                return Err(Error::Configuration(format!(
                    "Unsupported DuckDB distance function '{other}'."
                )))
            }
        },
    )
}

fn is_cosine(field: &VectorStoreField) -> bool {
    matches!(
        field.distance_function.as_ref().map(|d| d.as_str()),
        None | Some("DEFAULT" | "cosine_distance" | "cosine_similarity")
    )
}

/// A vector as a DuckDB array literal parameter (cast in SQL).
fn vector_param(field: &VectorStoreField, value: &Value) -> Result<Db> {
    let items = value.as_array().ok_or_else(|| {
        Error::Configuration(format!(
            "Vector field '{}' requires a dense numeric array.",
            field.name
        ))
    })?;
    let dims = field.dimensions.unwrap_or_default();
    if items.len() != dims {
        return Err(Error::Configuration(format!(
            "Vector field '{}' expects {dims} dimensions; got {}.",
            field.name,
            items.len()
        )));
    }
    let numbers = items
        .iter()
        .map(|i| i.as_f64().filter(|f| f.is_finite()))
        .collect::<Option<Vec<f64>>>()
        .ok_or_else(|| {
            Error::Configuration(format!(
                "Vector field '{}' requires finite numeric elements.",
                field.name
            ))
        })?;
    if is_cosine(field) && numbers.iter().all(|n| *n == 0.0) {
        return Err(Error::Configuration(format!(
            "Cosine vector field '{}' requires a nonzero vector.",
            field.name
        )));
    }
    Ok(Db::Text(serde_json::to_string(&numbers)?))
}

fn json_finite(value: &Value) -> bool {
    match value {
        Value::Number(n) => n.as_f64().is_some_and(f64::is_finite),
        Value::Array(items) => items.iter().all(json_finite),
        Value::Object(map) => map.values().all(json_finite),
        _ => true,
    }
}

/// A data or key value as a parameter, checked against the declared type.
fn value_param(field: &VectorStoreField, value: &Value) -> Result<Db> {
    if field.field_type == FieldType::Vector {
        if value.is_null() {
            return Ok(Db::Null);
        }
        return vector_param(field, value);
    }
    if value.is_null() {
        if field.field_type == FieldType::Key {
            return Err(Error::Configuration("DuckDB keys cannot be null.".into()));
        }
        return Ok(Db::Null);
    }
    let mismatch = || {
        Error::Configuration(format!(
            "Field '{}' requires a value of type '{}'.",
            field.name,
            field.type_.as_deref().unwrap_or_default()
        ))
    };
    Ok(match (field.type_.as_deref(), value) {
        (Some("str"), Value::String(s)) => Db::Text(s.clone()),
        (Some("int"), Value::Number(n)) => Db::BigInt(n.as_i64().ok_or_else(mismatch)?),
        (Some("float"), Value::Number(n)) => {
            Db::Double(n.as_f64().filter(|f| f.is_finite()).ok_or_else(mismatch)?)
        }
        (Some("bool"), Value::Bool(b)) => Db::Boolean(*b),
        (Some("UUID"), Value::String(s)) => Db::Text(
            uuid::Uuid::parse_str(s)
                .map_err(|_| mismatch())?
                .hyphenated()
                .to_string(),
        ),
        (Some("date"), Value::String(s)) => Db::Text(s.clone()),
        (Some("datetime"), Value::String(s)) => {
            let has_zone = s.ends_with('Z')
                || s.rfind(['+', '-'])
                    .is_some_and(|i| i > 10 && s[i..].contains(':'));
            if !has_zone {
                return Err(Error::Configuration(format!(
                    "Datetime field '{}' requires a timezone.",
                    field.name
                )));
            }
            Db::Text(s.clone())
        }
        (Some("list"), Value::Array(_)) | (Some("dict"), Value::Object(_)) => {
            if !json_finite(value) {
                return Err(Error::Configuration(
                    "JSON values must contain finite numbers.".into(),
                ));
            }
            Db::Text(value.to_string())
        }
        _ => return Err(mismatch()),
    })
}

/// A parameter placeholder cast to the column's type, so text-bound values
/// (UUIDs, dates, vectors) reach DuckDB typed.
fn placeholder(field: &VectorStoreField) -> Result<String> {
    Ok(match field.type_.as_deref() {
        Some("str") | Some("int") | Some("float") | Some("bool") | None
            if field.field_type != FieldType::Vector =>
        {
            "?".to_string()
        }
        _ => format!("CAST(? AS {})", column_type(field)?),
    })
}

/// Compiles portable filters into a parameterized predicate.
struct FilterCompiler<'a> {
    definition: &'a VectorStoreCollectionDefinition,
    params: Vec<Db>,
}

impl FilterCompiler<'_> {
    fn compile(mut self, expression: Option<&FilterExpression>) -> Result<(String, Vec<Db>)> {
        let sql = match expression {
            None => "TRUE".to_string(),
            Some(e) => self.condition(e)?,
        };
        Ok((sql, self.params))
    }

    fn equality(
        &mut self,
        field: &VectorStoreField,
        column: &str,
        value: &Value,
    ) -> Result<String> {
        if matches!(field.type_.as_deref(), Some("list" | "dict")) {
            return Err(Error::Configuration(
                "DuckDB JSON equality and collection filters are not supported.".into(),
            ));
        }
        if value.is_null() {
            return Ok(format!("{column} IS NULL"));
        }
        let param = match (field.type_.as_deref(), value) {
            (Some("int") | Some("float"), Value::Number(n)) => {
                Db::Double(n.as_f64().filter(|f| f.is_finite()).ok_or_else(|| {
                    Error::Configuration("Numeric filter operands must be finite.".into())
                })?)
            }
            _ => match value_param(field, value) {
                Ok(p) => p,
                // A value of the wrong type cannot equal anything.
                Err(_) => return Ok("FALSE".to_string()),
            },
        };
        self.params.push(param);
        Ok(format!(
            "{column} IS NOT DISTINCT FROM {}",
            placeholder(field)?
        ))
    }

    fn condition(&mut self, expression: &FilterExpression) -> Result<String> {
        let filter: &Filter = match expression {
            FilterExpression::Group(group) => {
                let parts = group
                    .filters
                    .iter()
                    .map(|c| self.condition(c))
                    .collect::<Result<Vec<_>>>()?;
                return Ok(match group.operator {
                    FilterGroupOperator::Not => format!("(NOT ({}))", parts[0]),
                    FilterGroupOperator::And => format!("({})", parts.join(" AND ")),
                    FilterGroupOperator::Or => format!("({})", parts.join(" OR ")),
                });
            }
            FilterExpression::Condition(f) => f,
        };
        if filter.field_name.contains('.') {
            return Err(Error::Configuration(
                "DuckDB does not support nested filter paths.".into(),
            ));
        }
        let field = self
            .definition
            .try_get_field(&filter.field_name)
            .ok_or_else(|| {
                Error::Configuration(format!("Unknown DuckDB field '{}'.", filter.field_name))
            })?;
        if field.field_type == FieldType::Vector {
            return Err(Error::Configuration(
                "Filtering vector columns is not supported.".into(),
            ));
        }
        let column = format!("t.{}", ident(field.effective_storage_name())?);
        let value = filter.value.clone().unwrap_or(Value::Null);
        Ok(match &filter.operator {
            FilterOperator::Exists => "TRUE".into(),
            FilterOperator::IsNull => format!("{column} IS NULL"),
            FilterOperator::IsNotNull => format!("{column} IS NOT NULL"),
            FilterOperator::Eq => self.equality(field, &column, &value)?,
            FilterOperator::Ne => format!("(NOT ({}))", self.equality(field, &column, &value)?),
            op @ (FilterOperator::In | FilterOperator::NotIn) => {
                let items = value.as_array().cloned().unwrap_or_default();
                let parts = items
                    .iter()
                    .map(|i| self.equality(field, &column, i))
                    .collect::<Result<Vec<_>>>()?;
                let choices = if parts.is_empty() {
                    "FALSE".to_string()
                } else {
                    format!("({})", parts.join(" OR "))
                };
                let membership = if *op == FilterOperator::In {
                    choices
                } else {
                    format!("(NOT {choices})")
                };
                format!("({column} IS NOT NULL AND {membership})")
            }
            op @ (FilterOperator::Gt
            | FilterOperator::Gte
            | FilterOperator::Lt
            | FilterOperator::Lte
            | FilterOperator::Between) => {
                if !matches!(
                    field.type_.as_deref(),
                    Some("str" | "int" | "float" | "date" | "datetime")
                ) {
                    return Err(Error::Configuration(format!(
                        "Ordered filtering is not supported for {:?}.",
                        field.type_
                    )));
                }
                let operands = if *op == FilterOperator::Between {
                    value.as_array().cloned().unwrap_or_default()
                } else {
                    vec![value.clone()]
                };
                for operand in &operands {
                    let param = match (field.type_.as_deref(), operand) {
                        (_, Value::Null | Value::Bool(_)) => {
                            return Err(Error::Configuration(
                                "Ordered filter operands must be non-null scalars of the column's type."
                                    .into(),
                            ))
                        }
                        (Some("int" | "float"), Value::Number(n)) => {
                            Db::Double(n.as_f64().filter(|f| f.is_finite()).ok_or_else(|| {
                                Error::Configuration("Ordered filter numbers must be finite.".into())
                            })?)
                        }
                        _ => value_param(field, operand)?,
                    };
                    self.params.push(param);
                }
                let ph = placeholder(field)?;
                match op {
                    FilterOperator::Between => {
                        format!("({column} BETWEEN {ph} AND {ph}) IS TRUE")
                    }
                    FilterOperator::Gt => format!("({column} > {ph}) IS TRUE"),
                    FilterOperator::Gte => format!("({column} >= {ph}) IS TRUE"),
                    FilterOperator::Lt => format!("({column} < {ph}) IS TRUE"),
                    _ => format!("({column} <= {ph}) IS TRUE"),
                }
            }
            op @ (FilterOperator::ContainsText
            | FilterOperator::StartsWith
            | FilterOperator::EndsWith) => {
                if field.type_.as_deref() != Some("str") {
                    return Err(Error::Configuration(
                        "Text filters require a string field.".into(),
                    ));
                }
                let text = value.as_str().ok_or_else(|| {
                    Error::Configuration("Text filters require a string operand.".into())
                })?;
                let escaped = text
                    .replace('!', "!!")
                    .replace('%', "!%")
                    .replace('_', "!_");
                let pattern = match op {
                    FilterOperator::StartsWith => format!("{escaped}%"),
                    FilterOperator::EndsWith => format!("%{escaped}"),
                    _ => format!("%{escaped}%"),
                };
                self.params.push(Db::Text(pattern));
                format!("({column} LIKE ? ESCAPE '!') IS TRUE")
            }
            other => {
                return Err(Error::Configuration(format!(
                    "Unsupported DuckDB filter operator '{other:?}'."
                )))
            }
        })
    }
}

/// A connection shared by a store's collections; statements run one at a
/// time on Tokio's blocking pool.
#[derive(Clone)]
struct Client {
    connection: Arc<Mutex<duckdb::Connection>>,
}

impl Client {
    async fn run<T, F>(&self, f: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&duckdb::Connection) -> Result<T> + Send + 'static,
    {
        let connection = self.connection.clone();
        tokio::task::spawn_blocking(move || {
            let guard = connection
                .lock()
                .map_err(|_| Error::service("the DuckDB connection is poisoned"))?;
            f(&guard)
        })
        .await
        .map_err(|e| Error::service(format!("DuckDB task failed: {e}")))?
    }
}

/// A table in a DuckDB database. See the [crate docs](crate).
#[derive(Clone)]
pub struct DuckDbCollection {
    client: Client,
    name: String,
    table: String,
    definition: VectorStoreCollectionDefinition,
}

impl std::fmt::Debug for DuckDbCollection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DuckDbCollection")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl DuckDbCollection {
    fn new(
        client: Client,
        name: &str,
        definition: VectorStoreCollectionDefinition,
    ) -> Result<Self> {
        let table = ident(name)?;
        let mut folded = std::collections::HashSet::new();
        for field in definition.fields() {
            let storage = field.effective_storage_name();
            ident(storage)?;
            if !folded.insert(storage.to_ascii_lowercase()) {
                return Err(Error::Configuration(
                    "DuckDB column names must be unique ignoring case.".into(),
                ));
            }
            column_type(field)?;
            if field.is_full_text_indexed == Some(true) || field.is_indexed == Some(true) {
                return Err(Error::Configuration(
                    "DuckDB full-text and explicit data indexes are not supported.".into(),
                ));
            }
            match field.field_type {
                FieldType::Vector => {
                    match field.index_kind.as_ref().map(|k| k.as_str()) {
                        None | Some("default" | "flat") => {}
                        Some(other) => {
                            return Err(Error::Configuration(format!(
                                "Unsupported DuckDB vector index kind '{other}'."
                            )))
                        }
                    }
                    metric(field)?;
                }
                FieldType::Key => {
                    if !matches!(field.type_.as_deref(), Some("str" | "int" | "UUID")) {
                        return Err(Error::Configuration(
                            "DuckDB keys must be str, int or UUID.".into(),
                        ));
                    }
                }
                FieldType::Data => {}
            }
        }
        Ok(Self {
            client,
            name: name.to_string(),
            table,
            definition,
        })
    }

    /// `json_object('name', t."col", …)` over the selected fields: one
    /// JSON text column carrying the record, whatever the column types.
    fn record_json(&self, include_vectors: bool) -> Result<String> {
        let mut pairs = Vec::new();
        for field in self.definition.fields() {
            if !include_vectors && field.field_type == FieldType::Vector {
                continue;
            }
            let name = field.effective_storage_name();
            pairs.push(format!("{}, t.{}", literal(name), ident(name)?));
        }
        Ok(format!(
            "CAST(json_object({}) AS VARCHAR)",
            pairs.join(", ")
        ))
    }

    fn parse_record(&self, text: &str) -> Result<Value> {
        let stored: Value = serde_json::from_str(text)
            .map_err(|e| Error::service(format!("DuckDB returned an unreadable row: {e}")))?;
        let mut stored = stored.as_object().cloned().unwrap_or_default();
        // JSON columns come back as JSON; keep explicit nulls out of the
        // record so it matches what was written.
        stored.retain(|_, v| !v.is_null());
        self.definition.from_storage(&Value::Object(stored), true)
    }

    /// The `SELECT … CROSS JOIN LATERAL … ORDER BY` statement for a search,
    /// and its parameters (test seam).
    pub fn search_sql(
        &self,
        vector: &[f32],
        options: &VectorSearchOptions,
    ) -> Result<(String, Vec<String>)> {
        let (sql, params) = self.search_statement(vector, options)?;
        Ok((sql, params.iter().map(|p| format!("{p:?}")).collect()))
    }

    fn search_statement(
        &self,
        vector: &[f32],
        options: &VectorSearchOptions,
    ) -> Result<(String, Vec<Db>)> {
        let field = self
            .definition
            .try_get_vector_field(options.vector_field_name.as_deref())
            .ok_or_else(|| {
                Error::Configuration("Select a vector field from the collection definition.".into())
            })?;
        let query = vector_param(
            field,
            &Value::Array(vector.iter().map(|v| serde_json::json!(*v)).collect()),
        )?;
        let (function, higher_is_closer) = metric(field)?;
        let column = format!("t.{}", ident(field.effective_storage_name())?);
        let mut distance = format!("{function}({column}, CAST(? AS {}))", column_type(field)?);
        if is_cosine(field) {
            // A zero stored vector has no cosine; it is skipped, not NaN.
            distance = format!(
                "CASE WHEN array_inner_product({column}, {column}) = 0 THEN NULL ELSE {distance} END"
            );
        }
        let (predicate, filter_params) = FilterCompiler {
            definition: &self.definition,
            params: Vec::new(),
        }
        .compile(options.filter.as_ref())?;
        let mut predicate = predicate;
        if let Some(raw) = options
            .provider_filter
            .as_deref()
            .filter(|r| !r.trim().is_empty())
        {
            predicate = format!("({predicate}) AND ({raw})");
        }
        let order = if higher_is_closer { "DESC" } else { "ASC" };
        let sql = format!(
            "SELECT {}, scores.value FROM {} AS t \
             CROSS JOIN LATERAL (SELECT {distance} AS value) AS scores \
             WHERE {column} IS NOT NULL AND ({predicate}) AND isfinite(scores.value) \
             ORDER BY scores.value {order}, t.{} ASC LIMIT ? OFFSET ?",
            self.record_json(options.include_vectors)?,
            self.table,
            ident(self.definition.key_field_storage_name())?,
        );
        let mut params = vec![query];
        params.extend(filter_params);
        params.push(Db::BigInt(options.top as i64));
        params.push(Db::BigInt(options.skip as i64));
        Ok((sql, params))
    }

    fn key_params(&self, keys: &[Value]) -> Result<Vec<Db>> {
        let key = self.definition.key_field();
        keys.iter().map(|k| value_param(key, k)).collect()
    }
}

fn exec(connection: &duckdb::Connection, sql: &str, params: &[Db]) -> Result<usize> {
    connection
        .execute(sql, duckdb::params_from_iter(params.iter()))
        .map_err(map_err)
}

const TABLE_SCOPE: &str = "FROM information_schema.tables \
     WHERE table_catalog = current_database() AND table_schema = current_schema() \
     AND table_type = 'BASE TABLE'";

#[async_trait]
impl VectorCollection for DuckDbCollection {
    fn name(&self) -> &str {
        &self.name
    }

    fn definition(&self) -> &VectorStoreCollectionDefinition {
        &self.definition
    }

    async fn ensure_collection_exists(&self) -> Result<()> {
        let columns = self
            .definition
            .fields()
            .iter()
            .map(|f| {
                Ok(format!(
                    "{} {}{}",
                    ident(f.effective_storage_name())?,
                    column_type(f)?,
                    if f.field_type == FieldType::Key {
                        " PRIMARY KEY"
                    } else {
                        ""
                    }
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let sql = format!(
            "CREATE TABLE IF NOT EXISTS {} ({})",
            self.table,
            columns.join(", ")
        );
        self.client
            .run(move |c| exec(c, &sql, &[]).map(|_| ()))
            .await
    }

    async fn collection_exists(&self) -> Result<bool> {
        let name = self.name.clone();
        self.client.run(move |c| table_exists(c, &name)).await
    }

    async fn ensure_collection_deleted(&self) -> Result<()> {
        let sql = format!("DROP TABLE IF EXISTS {}", self.table);
        self.client
            .run(move |c| exec(c, &sql, &[]).map(|_| ()))
            .await
    }

    async fn upsert(&self, records: Vec<Value>) -> Result<Vec<Value>> {
        if records.is_empty() {
            return Ok(Vec::new());
        }
        let key_name = self.definition.key_field_storage_name().to_string();
        let fields: Vec<VectorStoreField> = self.definition.fields().to_vec();
        let mut rows: Vec<Vec<Db>> = Vec::with_capacity(records.len());
        let mut keys = Vec::with_capacity(records.len());
        for record in &records {
            let stored = self.definition.to_storage(record)?;
            let key = stored.get(&key_name).cloned().ok_or_else(|| {
                Error::Configuration(format!(
                    "Record is missing vector store field '{}'.",
                    self.definition.key_field().name
                ))
            })?;
            keys.push(key);
            rows.push(
                fields
                    .iter()
                    .map(|f| {
                        value_param(
                            f,
                            stored
                                .get(f.effective_storage_name())
                                .unwrap_or(&Value::Null),
                        )
                    })
                    .collect::<Result<Vec<_>>>()?,
            );
        }
        let names = fields
            .iter()
            .map(|f| ident(f.effective_storage_name()))
            .collect::<Result<Vec<_>>>()?;
        let placeholders = fields.iter().map(placeholder).collect::<Result<Vec<_>>>()?;
        let key_ident = ident(&key_name)?;
        let updates: Vec<String> = names
            .iter()
            .filter(|n| **n != key_ident)
            .map(|n| format!("{n} = EXCLUDED.{n}"))
            .collect();
        let conflict = if updates.is_empty() {
            "DO NOTHING".to_string()
        } else {
            format!("DO UPDATE SET {}", updates.join(", "))
        };
        let sql = format!(
            "INSERT INTO {} ({}) VALUES ({}) ON CONFLICT ({key_ident}) {conflict}",
            self.table,
            names.join(", "),
            placeholders.join(", "),
        );
        self.client
            .run(move |c| {
                c.execute_batch("BEGIN TRANSACTION").map_err(map_err)?;
                for row in &rows {
                    if let Err(e) = exec(c, &sql, row) {
                        let _ = c.execute_batch("ROLLBACK");
                        return Err(e);
                    }
                }
                c.execute_batch("COMMIT").map_err(map_err)
            })
            .await?;
        Ok(keys)
    }

    async fn get(&self, keys: Vec<Value>, include_vectors: bool) -> Result<Vec<Option<Value>>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let params = self.key_params(&keys)?;
        let key_field = self.definition.key_field().clone();
        let key_storage = self.definition.key_field_storage_name().to_string();
        let sql = format!(
            "SELECT {} FROM {} AS t WHERE t.{} IN ({})",
            self.record_json(include_vectors)?,
            self.table,
            ident(&key_storage)?,
            vec![placeholder(&key_field)?; params.len()].join(", "),
        );
        let rows: Vec<String> = self
            .client
            .run(move |c| {
                let mut stmt = c.prepare(&sql).map_err(map_err)?;
                let rows = stmt
                    .query_map(duckdb::params_from_iter(params.iter()), |r| {
                        r.get::<_, String>(0)
                    })
                    .map_err(map_err)?;
                rows.collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(map_err)
            })
            .await?;
        let uuid_keys = key_field.type_.as_deref() == Some("UUID");
        let mut found = std::collections::HashMap::new();
        for row in rows {
            let record = self.parse_record(&row)?;
            let id = record
                .get(&key_field.name)
                .map(|k| normalized_key(k, uuid_keys))
                .unwrap_or_default();
            found.insert(id, record);
        }
        Ok(keys
            .iter()
            .map(|k| found.get(&normalized_key(k, uuid_keys)).cloned())
            .collect())
    }

    async fn delete(&self, keys: Vec<Value>) -> Result<()> {
        if keys.is_empty() {
            return Ok(());
        }
        let params = self.key_params(&keys)?;
        let sql = format!(
            "DELETE FROM {} WHERE {} IN ({})",
            self.table,
            ident(self.definition.key_field_storage_name())?,
            vec![placeholder(self.definition.key_field())?; params.len()].join(", "),
        );
        self.client
            .run(move |c| exec(c, &sql, &params).map(|_| ()))
            .await
    }

    async fn search(
        &self,
        vector: Vec<f32>,
        options: &VectorSearchOptions,
    ) -> Result<Vec<VectorSearchResult>> {
        options.validate()?;
        let (sql, params) = self.search_statement(&vector, options)?;
        let rows: Vec<(String, f64)> = self
            .client
            .run(move |c| {
                let mut stmt = c.prepare(&sql).map_err(map_err)?;
                let rows = stmt
                    .query_map(duckdb::params_from_iter(params.iter()), |r| {
                        Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)?))
                    })
                    .map_err(map_err)?;
                rows.collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(map_err)
            })
            .await?;
        rows.into_iter()
            .map(|(json, score)| {
                Ok(VectorSearchResult {
                    record: self.parse_record(&json)?,
                    score: Some(score),
                    score_kind: None,
                })
            })
            .collect()
    }
}

/// A key compared as text. UUID-typed keys are canonicalized (DuckDB stores
/// them hyphenated and lowercase, so `"A1B2…"` finds `"a1b2…"`); string keys
/// are compared exactly, so distinct UUID-shaped strings stay distinct.
fn normalized_key(value: &Value, uuid: bool) -> String {
    match value {
        Value::String(s) if !uuid => s.clone(),
        Value::String(s) => uuid::Uuid::parse_str(s)
            .map(|u| u.hyphenated().to_string())
            .unwrap_or_else(|_| s.clone()),
        other => other.to_string(),
    }
}

fn table_exists(connection: &duckdb::Connection, name: &str) -> Result<bool> {
    let sql = format!("SELECT count(*) {TABLE_SCOPE} AND lower(table_name) = lower(?)");
    let count: i64 = connection
        .query_row(&sql, [name], |r| r.get(0))
        .map_err(map_err)?;
    Ok(count > 0)
}

/// A DuckDB database holding vector collections. See the [crate docs](crate).
#[derive(Clone)]
pub struct DuckDbStore {
    client: Client,
}

impl std::fmt::Debug for DuckDbStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DuckDbStore").finish_non_exhaustive()
    }
}

impl DuckDbStore {
    /// Open (or create) the database at `path`.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Ok(Self::from_connection(
            duckdb::Connection::open(path).map_err(map_err)?,
        ))
    }

    /// An in-memory database, discarded when the store is dropped.
    pub fn in_memory() -> Result<Self> {
        Ok(Self::from_connection(
            duckdb::Connection::open_in_memory().map_err(map_err)?,
        ))
    }

    /// From `DUCKDB_CONNECTION_STRING`, else [`DEFAULT_DATABASE`], as
    /// upstream.
    pub fn from_env() -> Result<Self> {
        let path = std::env::var("DUCKDB_CONNECTION_STRING")
            .ok()
            .filter(|p| !p.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_DATABASE.to_string());
        Self::open(path)
    }

    /// Use an existing connection.
    pub fn from_connection(connection: duckdb::Connection) -> Self {
        Self {
            client: Client {
                connection: Arc::new(Mutex::new(connection)),
            },
        }
    }

    /// A typed collection handle.
    pub fn collection(
        &self,
        name: &str,
        definition: VectorStoreCollectionDefinition,
    ) -> Result<DuckDbCollection> {
        DuckDbCollection::new(self.client.clone(), name, definition)
    }
}

#[async_trait]
impl VectorStore for DuckDbStore {
    fn get_collection(
        &self,
        name: &str,
        definition: VectorStoreCollectionDefinition,
    ) -> Result<Box<dyn VectorCollection>> {
        Ok(Box::new(self.collection(name, definition)?))
    }

    async fn list_collection_names(&self) -> Result<Vec<String>> {
        self.client
            .run(|c| {
                let mut stmt = c
                    .prepare(&format!(
                        "SELECT table_name {TABLE_SCOPE} ORDER BY table_name"
                    ))
                    .map_err(map_err)?;
                let rows = stmt
                    .query_map([], |r| r.get::<_, String>(0))
                    .map_err(map_err)?;
                rows.collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(map_err)
            })
            .await
    }

    async fn collection_exists(&self, name: &str) -> Result<bool> {
        let name = name.to_string();
        self.client.run(move |c| table_exists(c, &name)).await
    }
}
