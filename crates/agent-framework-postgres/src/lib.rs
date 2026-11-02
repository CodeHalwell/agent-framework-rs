//! A PostgreSQL + pgvector [`VectorStore`]: one table per collection, typed
//! columns, and dense-vector search.
//!
//! Port of upstream `agent_framework_postgres` (`PostgresCollection`,
//! `PostgresStore`) over `tokio-postgres`.
//!
//! ```no_run
//! use agent_framework_core::vectors::{
//!     Filter, VectorSearchOptions, VectorStore, VectorStoreCollectionDefinition, VectorStoreField,
//! };
//! use agent_framework_postgres::PostgresStore;
//!
//! # async fn demo() -> agent_framework_core::error::Result<()> {
//! let store = PostgresStore::connect("host=localhost user=postgres").await?;
//! let definition = VectorStoreCollectionDefinition::new(vec![
//!     VectorStoreField::key("id").with_type("int"),
//!     VectorStoreField::data("year").with_type("int").indexed(),
//!     VectorStoreField::vector("embedding", 3),
//! ])?;
//! let docs = store.get_collection("docs", definition)?;
//! docs.ensure_collection_exists().await?;
//! let hits = docs
//!     .search(vec![0.1, 0.2, 0.3], &VectorSearchOptions::new(5).with_filter(Filter::gte("year", 2020)?))
//!     .await?;
//! # let _ = hits;
//! # Ok(())
//! # }
//! ```
//!
//! # Model
//!
//! - The schema and the `vector` extension must already exist; the store
//!   never creates extensions or schemas, and never migrates a table.
//! - Every non-vector field needs a declared type: `str` (`text COLLATE "C"`),
//!   `int` (`bigint`), `float`, `bool`, `UUID`, `bytes` (`bytea`, as a JSON
//!   array of byte values), `date`, `datetime` (`timestamptz`; a timezone is
//!   required), `list` / `dict` (`jsonb`). Keys are `str`, `int` or `UUID`.
//! - Vector fields are `vector(n)`, or `halfvec(n)` for `float16`. Metrics:
//!   cosine (default; `cosine_similarity` scores `1 - distance`), `dot_prod`
//!   (scored as the dot product), `euclidean_distance`, `manhattan`. Index
//!   kinds: default/flat (exact search), `hnsw` (m 16, ef_construction 64)
//!   and `ivf_flat` (100 lists; needs rows before the index is built, see
//!   [`PostgresCollection::ensure_collection_exists_with`]).
//! - Portable filters compile to parameterized SQL with total boolean
//!   semantics: a type-mismatched equality is `FALSE` rather than a coercion,
//!   and text operators escape `%` and `_`.
//! - Values travel as text with explicit casts, so the server checks every
//!   value against its column type.
//! - [`VectorSearchOptions::provider_filter`] is a raw SQL boolean
//!   expression, conjoined with the portable filter. It is interpolated as
//!   given: never build it from untrusted input.
//!
//! # Divergences
//!
//! - One `tokio-postgres` client (pipelined) instead of a connection pool;
//!   batched writes are single statements, so each is atomic.
//! - Upstream's `postgres.*` field annotations (index tuning, `halfvec`
//!   override) have no counterpart: the Rust field definition has no
//!   provider annotations, so the defaults above apply.
//! - No search operation options (`exact`, `hnsw_ef_search`,
//!   `ivfflat_probes`) or score threshold: the trait has no slot for them.

#![cfg(feature = "experimental-vector-stores")]
#![cfg_attr(docsrs, doc(cfg(feature = "experimental-vector-stores")))]

use std::sync::Arc;

use agent_framework_core::error::{Error, Result};
use agent_framework_core::vectors::{
    FieldType, Filter, FilterExpression, FilterGroupOperator, FilterOperator, VectorCollection,
    VectorSearchOptions, VectorSearchResult, VectorStore, VectorStoreCollectionDefinition,
    VectorStoreField,
};
use async_trait::async_trait;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use tokio_postgres::types::ToSql;

fn db_err(e: tokio_postgres::Error) -> Error {
    let detail = e
        .as_db_error()
        .map(|d| d.message().to_string())
        .unwrap_or_else(|| e.to_string());
    Error::Service(format!("PostgreSQL operation failed: {detail}"))
}

fn config(message: impl Into<String>) -> Error {
    Error::Configuration(message.into())
}

/// Quote one identifier, refusing truncation and NUL.
pub fn quote_identifier(name: &str) -> Result<String> {
    if name.is_empty() || name.contains('\0') || name.len() > 63 {
        return Err(config(
            "Postgres identifiers must contain 1-63 UTF-8 bytes and no NUL.",
        ));
    }
    Ok(format!("\"{}\"", name.replace('"', "\"\"")))
}

/// The SQL type of a non-vector column, without collation.
fn base_type(field: &VectorStoreField) -> Result<&'static str> {
    Ok(match field.type_.as_deref() {
        Some("str") => "text",
        Some("int") => "bigint",
        Some("float") => "double precision",
        Some("bool") => "boolean",
        Some("UUID") => "uuid",
        Some("bytes") => "bytea",
        Some("date") => "date",
        Some("datetime") => "timestamp with time zone",
        Some("list" | "dict") => "jsonb",
        other => {
            return Err(config(format!(
                "Field '{}' needs a supported explicit type; got '{}'.",
                field.name,
                other.unwrap_or("None")
            )))
        }
    })
}

fn vector_type(field: &VectorStoreField) -> &'static str {
    if field.type_.as_deref() == Some("float16") {
        "halfvec"
    } else {
        "vector"
    }
}

/// `(operator, opclass suffix)` for a vector field's metric.
fn metric(field: &VectorStoreField) -> Result<(&'static str, &'static str)> {
    Ok(
        match field
            .distance_function
            .as_ref()
            .map(|d| d.as_str())
            .unwrap_or("DEFAULT")
        {
            "DEFAULT" | "cosine_distance" | "cosine_similarity" => ("<=>", "cosine"),
            "dot_prod" | "negative_dot_prod" => ("<#>", "ip"),
            "euclidean_distance" => ("<->", "l2"),
            "manhattan" => ("<+>", "l1"),
            other => {
                return Err(config(format!(
                    "Unsupported Postgres distance function '{other}'."
                )))
            }
        },
    )
}

fn column_type(field: &VectorStoreField) -> Result<String> {
    if field.field_type == FieldType::Vector {
        return Ok(format!(
            "{}({})",
            vector_type(field),
            field.dimensions.unwrap_or_default()
        ));
    }
    Ok(match base_type(field)? {
        "text" => "text COLLATE \"C\"".to_string(),
        other => other.to_string(),
    })
}

/// The cast a text parameter takes to reach `field`'s column type.
fn cast_type(field: &VectorStoreField) -> Result<String> {
    if field.field_type == FieldType::Vector {
        return column_type(field);
    }
    base_type(field).map(str::to_string)
}

fn index_kind(field: &VectorStoreField) -> &str {
    field
        .index_kind
        .as_ref()
        .map(|k| k.as_str())
        .unwrap_or("default")
}

/// A vector rendered as pgvector text.
fn vector_text(value: &Value, name: &str) -> Result<String> {
    let Value::Array(items) = value else {
        return Err(config(format!(
            "Vector field '{name}' requires a dense numeric vector."
        )));
    };
    let mut parts = Vec::with_capacity(items.len());
    for item in items {
        let n = item
            .as_f64()
            .filter(|n| n.is_finite() && !item.is_boolean())
            .ok_or_else(|| config(format!("Vector field '{name}' requires finite numbers.")))?;
        parts.push(n.to_string());
    }
    Ok(format!("[{}]", parts.join(",")))
}

/// `value` as the text the column's cast accepts, after checking its JSON
/// type against the field's declared type.
fn value_text(field: &VectorStoreField, value: &Value) -> Result<Option<String>> {
    if value.is_null() {
        return Ok(None);
    }
    if field.field_type == FieldType::Vector {
        return vector_text(value, &field.name).map(Some);
    }
    let mismatch = || {
        config(format!(
            "Field '{}' requires a value of type '{}'.",
            field.name,
            field.type_.as_deref().unwrap_or("None")
        ))
    };
    let text = match (field.type_.as_deref(), value) {
        (Some("str"), Value::String(s)) => s.clone(),
        (Some("int"), Value::Number(n)) if n.is_i64() => n.to_string(),
        (Some("float"), Value::Number(n)) => {
            let f = n.as_f64().filter(|f| f.is_finite()).ok_or_else(mismatch)?;
            f.to_string()
        }
        (Some("bool"), Value::Bool(b)) => b.to_string(),
        (Some("UUID"), Value::String(s)) => uuid::Uuid::parse_str(s)
            .map_err(|_| mismatch())?
            .hyphenated()
            .to_string(),
        (Some("bytes"), Value::Array(items)) => {
            let mut hex = String::from("\\x");
            for item in items {
                let byte = item.as_u64().filter(|b| *b <= 255).ok_or_else(mismatch)?;
                hex.push_str(&format!("{byte:02x}"));
            }
            hex
        }
        (Some("date"), Value::String(s)) => s.clone(),
        (Some("datetime"), Value::String(s)) => {
            if !has_timezone(s) {
                return Err(config(format!(
                    "Datetime field '{}' requires a timezone.",
                    field.name
                )));
            }
            s.clone()
        }
        (Some("list"), Value::Array(_)) | (Some("dict"), Value::Object(_)) => value.to_string(),
        _ => return Err(mismatch()),
    };
    Ok(Some(text))
}

fn has_timezone(s: &str) -> bool {
    let Some(t) = s.find(['T', ' ']) else {
        return false;
    };
    let time = &s[t + 1..];
    time.ends_with('Z') || time.ends_with('z') || time.contains('+') || time.contains('-')
}

/// A column's text (as `SELECT col::text` returns it) as the record value.
fn read_value(field: &VectorStoreField, text: Option<String>) -> Value {
    let Some(text) = text else {
        return Value::Null;
    };
    if field.field_type == FieldType::Vector {
        // pgvector prints whole numbers without a fraction; vectors are floats.
        return match serde_json::from_str::<Vec<f64>>(&text) {
            Ok(items) => Value::Array(
                items
                    .into_iter()
                    .filter_map(serde_json::Number::from_f64)
                    .map(Value::Number)
                    .collect(),
            ),
            Err(_) => Value::String(text),
        };
    }
    match field.type_.as_deref() {
        Some("int") => text
            .parse::<i64>()
            .map(Value::from)
            .unwrap_or(Value::String(text)),
        Some("float") => text
            .parse::<f64>()
            .ok()
            .and_then(serde_json::Number::from_f64)
            .map(Value::Number)
            .unwrap_or(Value::String(text)),
        Some("bool") => Value::Bool(text == "t" || text == "true"),
        Some("list" | "dict") => serde_json::from_str(&text).unwrap_or(Value::String(text)),
        Some("bytes") => {
            let hex = text.trim_start_matches("\\x");
            Value::Array(
                (0..hex.len() / 2)
                    .filter_map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).ok())
                    .map(Value::from)
                    .collect(),
            )
        }
        _ => Value::String(text),
    }
}

/// Builds a parameterized statement; every parameter is text.
#[derive(Default)]
struct Params(Vec<Option<String>>);

impl Params {
    /// A placeholder for `value`, cast to `ty`.
    fn push(&mut self, value: Option<String>, ty: &str) -> String {
        self.0.push(value);
        format!("${}::text::{ty}", self.0.len())
    }

    fn refs(&self) -> Vec<&(dyn ToSql + Sync)> {
        self.0.iter().map(|p| p as &(dyn ToSql + Sync)).collect()
    }
}

/// Compiles a portable filter into SQL over a definition's columns.
struct FilterCompiler<'a> {
    definition: &'a VectorStoreCollectionDefinition,
    params: &'a mut Params,
}

impl FilterCompiler<'_> {
    fn field(&self, name: &str) -> Result<&VectorStoreField> {
        if name.contains('.') {
            return Err(config(
                "Postgres filters do not support nested field paths.",
            ));
        }
        let field = self
            .definition
            .try_get_field(name)
            .ok_or_else(|| config(format!("Unknown Postgres field '{name}'.")))?;
        if field.field_type == FieldType::Vector {
            return Err(config("Filtering vector columns is not supported."));
        }
        Ok(field)
    }

    fn compile(&mut self, expression: &FilterExpression) -> Result<String> {
        match expression {
            FilterExpression::Group(group) => {
                let parts = group
                    .filters
                    .iter()
                    .map(|c| self.compile(c))
                    .collect::<Result<Vec<_>>>()?;
                Ok(match group.operator {
                    FilterGroupOperator::Not => format!("(NOT ({}))", parts[0]),
                    FilterGroupOperator::And => format!("({})", parts.join(" AND ")),
                    FilterGroupOperator::Or => format!("({})", parts.join(" OR ")),
                })
            }
            FilterExpression::Condition(filter) => self.condition(filter),
        }
    }

    fn equality(
        &mut self,
        field: &VectorStoreField,
        column: &str,
        value: &Value,
    ) -> Result<String> {
        if value.is_null() {
            return Ok(format!("{column} IS NULL"));
        }
        let kind = field.type_.as_deref();
        let matches = match (kind, value) {
            (Some("bool"), Value::Bool(_)) => true,
            (Some("bool"), _) | (_, Value::Bool(_)) => false,
            (Some("int" | "float"), Value::Number(n)) => {
                if !n.as_f64().is_some_and(f64::is_finite) {
                    return Err(config("Numeric filter values must be finite."));
                }
                true
            }
            (Some("str" | "UUID" | "date" | "datetime"), Value::String(_)) => true,
            (Some("list" | "bytes"), Value::Array(_)) | (Some("dict"), Value::Object(_)) => true,
            _ => false,
        };
        if !matches {
            return Ok("FALSE".into());
        }
        let (text, ty) = match (kind, value) {
            // Compare numerically, so 2 matches 2.0 in either column type.
            (Some("int" | "float"), Value::Number(n)) => {
                (Some(n.to_string()), "numeric".to_string())
            }
            _ => (value_text(field, value)?, cast_type(field)?),
        };
        let placeholder = self.params.push(text, &ty);
        Ok(format!("{column} IS NOT DISTINCT FROM {placeholder}"))
    }

    fn condition(&mut self, filter: &Filter) -> Result<String> {
        let field = self.field(&filter.field_name)?.clone();
        let mut column = quote_identifier(field.effective_storage_name())?;
        if field.type_.as_deref() == Some("str") {
            column = format!("{column} COLLATE \"C\"");
        }
        let value = filter.value.clone().unwrap_or(Value::Null);
        let kind = field.type_.as_deref();
        Ok(match &filter.operator {
            // Declared columns exist in every row, even when NULL.
            FilterOperator::Exists => "TRUE".into(),
            FilterOperator::IsNull => format!("{column} IS NULL"),
            FilterOperator::IsNotNull => format!("{column} IS NOT NULL"),
            FilterOperator::Eq => self.equality(&field, &column, &value)?,
            FilterOperator::Ne => format!("(NOT ({}))", self.equality(&field, &column, &value)?),
            op @ (FilterOperator::In | FilterOperator::NotIn) => {
                let items = value.as_array().cloned().unwrap_or_default();
                let parts = items
                    .iter()
                    .map(|i| self.equality(&field, &column, i))
                    .collect::<Result<Vec<_>>>()?;
                let choices = if parts.is_empty() {
                    "FALSE".to_string()
                } else {
                    parts.join(" OR ")
                };
                let membership = if *op == FilterOperator::NotIn {
                    format!("(NOT ({choices}))")
                } else {
                    format!("({choices})")
                };
                // Unlike eq/ne, membership never matches a NULL field.
                format!("({column} IS NOT NULL AND {membership})")
            }
            op @ (FilterOperator::Gt
            | FilterOperator::Gte
            | FilterOperator::Lt
            | FilterOperator::Lte
            | FilterOperator::Between) => {
                if matches!(kind, Some("list" | "dict" | "bool" | "bytes")) {
                    return Err(config(format!(
                        "Ordered filtering is not supported for '{}'.",
                        kind.unwrap_or_default()
                    )));
                }
                let operands = if *op == FilterOperator::Between {
                    value.as_array().cloned().unwrap_or_default()
                } else {
                    vec![value.clone()]
                };
                let mut placeholders = Vec::new();
                for operand in &operands {
                    if operand.is_null() || operand.is_boolean() {
                        return Err(config(
                            "Ordered filter operands must be non-null scalars of the column's type.",
                        ));
                    }
                    let (text, ty) = match (kind, operand) {
                        (Some("int" | "float"), Value::Number(n)) => {
                            if !n.as_f64().is_some_and(f64::is_finite) {
                                return Err(config("Ordered filter numbers must be finite."));
                            }
                            (Some(n.to_string()), "numeric".to_string())
                        }
                        _ => (value_text(&field, operand)?, cast_type(&field)?),
                    };
                    placeholders.push(self.params.push(text, &ty));
                }
                let comparison = match op {
                    FilterOperator::Between => format!(
                        "{column} BETWEEN {} AND {}",
                        placeholders[0], placeholders[1]
                    ),
                    FilterOperator::Gt => format!("{column} > {}", placeholders[0]),
                    FilterOperator::Gte => format!("{column} >= {}", placeholders[0]),
                    FilterOperator::Lt => format!("{column} < {}", placeholders[0]),
                    _ => format!("{column} <= {}", placeholders[0]),
                };
                format!("({comparison}) IS TRUE")
            }
            op @ (FilterOperator::ContainsText
            | FilterOperator::StartsWith
            | FilterOperator::EndsWith) => {
                if kind != Some("str") {
                    return Err(config(format!(
                        "Text filtering requires a string column, not '{}'.",
                        kind.unwrap_or("None")
                    )));
                }
                let Value::String(text) = &value else {
                    return Err(config("Text filtering requires a string operand."));
                };
                // `!` is the escape character; `%` and `_` stay literal.
                let escaped = text
                    .replace('!', "!!")
                    .replace('%', "!%")
                    .replace('_', "!_");
                let pattern = match op {
                    FilterOperator::StartsWith => format!("{escaped}%"),
                    FilterOperator::EndsWith => format!("%{escaped}"),
                    _ => format!("%{escaped}%"),
                };
                let placeholder = self.params.push(Some(pattern), "text");
                format!("({column} LIKE {placeholder} ESCAPE '!') IS TRUE")
            }
            op @ (FilterOperator::Contains
            | FilterOperator::ContainsAny
            | FilterOperator::ContainsAll) => {
                if kind != Some("list") {
                    return Err(config("Collection membership requires a list column."));
                }
                let operands = if *op == FilterOperator::Contains {
                    vec![value.clone()]
                } else {
                    value.as_array().cloned().unwrap_or_default()
                };
                let parts: Vec<String> = operands
                    .iter()
                    .map(|item| {
                        let placeholder = self.params.push(Some(item.to_string()), "jsonb");
                        format!(
                            "EXISTS (SELECT 1 FROM jsonb_array_elements({column}) AS item(value) WHERE item.value = {placeholder})"
                        )
                    })
                    .collect();
                let all = *op == FilterOperator::ContainsAll;
                let combined = if parts.is_empty() {
                    if all { "TRUE" } else { "FALSE" }.to_string()
                } else {
                    parts.join(if all { " AND " } else { " OR " })
                };
                format!("({column} IS NOT NULL AND ({combined}))")
            }
            other => {
                return Err(config(format!(
                    "Unsupported Postgres filter operator '{other:?}'."
                )))
            }
        })
    }
}

/// A table in a PostgreSQL schema. See the [crate docs](crate).
#[derive(Clone)]
pub struct PostgresCollection {
    client: Arc<tokio_postgres::Client>,
    schema: String,
    name: String,
    table: String,
    definition: VectorStoreCollectionDefinition,
}

impl std::fmt::Debug for PostgresCollection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PostgresCollection")
            .field("table", &self.table)
            .finish_non_exhaustive()
    }
}

impl PostgresCollection {
    fn new(
        client: Arc<tokio_postgres::Client>,
        schema: &str,
        name: &str,
        definition: VectorStoreCollectionDefinition,
    ) -> Result<Self> {
        let table = format!("{}.{}", quote_identifier(schema)?, quote_identifier(name)?);
        for field in definition.fields() {
            quote_identifier(field.effective_storage_name())?;
            if field.is_full_text_indexed == Some(true) {
                return Err(config(
                    "Postgres full-text indexes and keyword-hybrid search are not supported.",
                ));
            }
            match field.field_type {
                FieldType::Key => {
                    if !matches!(field.type_.as_deref(), Some("str" | "int" | "UUID")) {
                        return Err(config("Postgres keys must be declared str, int or UUID."));
                    }
                }
                FieldType::Data => {
                    base_type(field)?;
                }
                FieldType::Vector => {
                    let dims = field.dimensions.unwrap_or_default();
                    if !(1..=16000).contains(&dims) {
                        return Err(config("dimensions must be an integer between 1 and 16000."));
                    }
                    let (_, ops) = metric(field)?;
                    let kind = index_kind(field);
                    if !matches!(kind, "default" | "flat" | "hnsw" | "ivf_flat") {
                        return Err(config(format!("Unsupported Postgres index kind '{kind}'.")));
                    }
                    if kind == "ivf_flat" && ops == "l1" {
                        return Err(config("IVFFlat does not support Manhattan distance."));
                    }
                    let limit = if vector_type(field) == "halfvec" {
                        4000
                    } else {
                        2000
                    };
                    if matches!(kind, "hnsw" | "ivf_flat") && dims > limit {
                        return Err(config(format!(
                            "indexed vector dimensions must be an integer between 1 and {limit}."
                        )));
                    }
                }
            }
        }
        Ok(Self {
            client,
            schema: schema.to_string(),
            name: name.to_string(),
            table,
            definition,
        })
    }

    /// The `CREATE INDEX` statement for an indexed data field or an ANN
    /// vector field.
    fn index_statement(&self, field: &VectorStoreField) -> Result<String> {
        let storage = field.effective_storage_name();
        let digest =
            Sha256::digest(format!("{}\0{}\0{storage}", self.schema, self.name).as_bytes());
        let hex: String = digest.iter().take(16).map(|b| format!("{b:02x}")).collect();
        let name = quote_identifier(&format!("af_vector_{hex}"))?;
        let column = quote_identifier(storage)?;
        if field.field_type != FieldType::Vector {
            return Ok(format!(
                "CREATE INDEX IF NOT EXISTS {name} ON {} ({column})",
                self.table
            ));
        }
        let (_, ops) = metric(field)?;
        let opclass = quote_identifier(&format!("{}_{ops}_ops", vector_type(field)))?;
        Ok(if index_kind(field) == "hnsw" {
            format!(
                "CREATE INDEX IF NOT EXISTS {name} ON {} USING hnsw ({column} {opclass}) WITH (m = 16, ef_construction = 64)",
                self.table
            )
        } else {
            format!(
                "CREATE INDEX IF NOT EXISTS {name} ON {} USING ivfflat ({column} {opclass}) WITH (lists = 100)",
                self.table
            )
        })
    }

    /// Create the table and, with `create_indexes`, its indexes, without
    /// altering an existing table. IVFFlat indexes need rows to train on:
    /// create with `false`, upsert, then call again with `true`.
    pub async fn ensure_collection_exists_with(&self, create_indexes: bool) -> Result<()> {
        let mut columns = Vec::new();
        for field in self.definition.fields() {
            let suffix = if field.field_type == FieldType::Key {
                " PRIMARY KEY"
            } else {
                ""
            };
            columns.push(format!(
                "{} {}{suffix}",
                quote_identifier(field.effective_storage_name())?,
                column_type(field)?
            ));
        }
        self.client
            .batch_execute(&format!(
                "CREATE TABLE IF NOT EXISTS {} ({})",
                self.table,
                columns.join(", ")
            ))
            .await
            .map_err(db_err)?;
        if !create_indexes {
            return Ok(());
        }
        for field in self.definition.fields() {
            let wanted = match field.field_type {
                FieldType::Vector => matches!(index_kind(field), "hnsw" | "ivf_flat"),
                FieldType::Data => field.is_indexed == Some(true),
                FieldType::Key => false,
            };
            if !wanted {
                continue;
            }
            if field.field_type == FieldType::Vector && index_kind(field) == "ivf_flat" {
                let row = self
                    .client
                    .query_opt(
                        &format!(
                            "SELECT 1 FROM {} WHERE {} IS NOT NULL LIMIT 1",
                            self.table,
                            quote_identifier(field.effective_storage_name())?
                        ),
                        &[],
                    )
                    .await
                    .map_err(db_err)?;
                if row.is_none() {
                    return Err(config(
                        "IVFFlat requires training data. Create with create_indexes=false, upsert \
                         vectors, then call ensure_collection_exists again.",
                    ));
                }
            }
            self.client
                .batch_execute(&self.index_statement(field)?)
                .await
                .map_err(db_err)?;
        }
        Ok(())
    }

    /// The `SELECT` list (each column as text) and the fields in that order.
    fn select_list(&self, include_vectors: bool) -> Result<(String, Vec<&VectorStoreField>)> {
        let fields: Vec<&VectorStoreField> = self
            .definition
            .fields()
            .iter()
            .filter(|f| include_vectors || f.field_type != FieldType::Vector)
            .collect();
        let list = fields
            .iter()
            .map(|f| {
                Ok(format!(
                    "{}::text",
                    quote_identifier(f.effective_storage_name())?
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((list.join(", "), fields))
    }

    fn row_record(fields: &[&VectorStoreField], row: &tokio_postgres::Row) -> Value {
        let mut record = Map::new();
        for (i, field) in fields.iter().enumerate() {
            let text: Option<String> = row.get(i);
            record.insert(
                field.effective_storage_name().to_string(),
                read_value(field, text),
            );
        }
        Value::Object(record)
    }

    fn key_texts(&self, keys: &[Value]) -> Result<Vec<String>> {
        let key = self.definition.key_field();
        keys.iter()
            .map(|k| value_text(key, k)?.ok_or_else(|| config("Postgres keys cannot be null.")))
            .collect()
    }

    /// The WHERE clause for `options`' filters.
    pub fn prepare_filter(
        &self,
        options: &VectorSearchOptions,
    ) -> Result<(String, Vec<Option<String>>)> {
        let mut params = Params::default();
        let sql = self.filter_sql(options, &mut params)?;
        Ok((sql, params.0))
    }

    fn filter_sql(&self, options: &VectorSearchOptions, params: &mut Params) -> Result<String> {
        let mut parts = Vec::new();
        if let Some(filter) = &options.filter {
            parts.push(
                FilterCompiler {
                    definition: &self.definition,
                    params,
                }
                .compile(filter)?,
            );
        }
        if let Some(raw) = &options.provider_filter {
            parts.push(format!("({raw})"));
        }
        Ok(if parts.is_empty() {
            "TRUE".into()
        } else {
            parts.join(" AND ")
        })
    }
}

#[async_trait]
impl VectorCollection for PostgresCollection {
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
        let row = self
            .client
            .query_opt(
                "SELECT 1 FROM information_schema.tables WHERE table_schema = $1 AND table_name = $2 \
                 AND table_type = 'BASE TABLE'",
                &[&self.schema, &self.name],
            )
            .await
            .map_err(db_err)?;
        Ok(row.is_some())
    }

    async fn ensure_collection_deleted(&self) -> Result<()> {
        self.client
            .batch_execute(&format!("DROP TABLE IF EXISTS {}", self.table))
            .await
            .map_err(db_err)
    }

    async fn upsert(&self, records: Vec<Value>) -> Result<Vec<Value>> {
        if records.is_empty() {
            return Ok(Vec::new());
        }
        let fields = self.definition.fields();
        let key = self.definition.key_field();
        let key_name = quote_identifier(key.effective_storage_name())?;
        let names = fields
            .iter()
            .map(|f| quote_identifier(f.effective_storage_name()))
            .collect::<Result<Vec<_>>>()?;
        let mut params = Params::default();
        let mut rows = Vec::new();
        for record in &records {
            let mut placeholders = Vec::new();
            for field in fields {
                let value = record
                    .get(field.effective_storage_name())
                    .unwrap_or(&Value::Null);
                if field.field_type == FieldType::Key && value.is_null() {
                    return Err(config("Postgres keys cannot be null."));
                }
                placeholders.push(params.push(value_text(field, value)?, &cast_type(field)?));
            }
            rows.push(format!("({})", placeholders.join(", ")));
        }
        let updates: Vec<String> = fields
            .iter()
            .zip(&names)
            .filter(|(f, _)| f.field_type != FieldType::Key)
            .map(|(_, n)| format!("{n} = EXCLUDED.{n}"))
            .collect();
        let conflict = if updates.is_empty() {
            format!("DO UPDATE SET {key_name} = EXCLUDED.{key_name}")
        } else {
            format!("DO UPDATE SET {}", updates.join(", "))
        };
        // One statement, so the batch is atomic. A key repeated in one batch
        // would make ON CONFLICT touch a row twice, which Postgres refuses,
        // so keep the last occurrence, as sequential upserts would.
        let statement = format!(
            "INSERT INTO {} ({}) VALUES {} ON CONFLICT ({key_name}) {conflict} RETURNING {key_name}::text",
            self.table,
            names.join(", "),
            rows.join(", ")
        );
        let key_texts = self.key_texts(
            &records
                .iter()
                .map(|r| {
                    r.get(key.effective_storage_name())
                        .cloned()
                        .unwrap_or(Value::Null)
                })
                .collect::<Vec<_>>(),
        )?;
        let mut seen = std::collections::HashSet::new();
        if key_texts.iter().any(|k| !seen.insert(k.clone())) {
            for record in records.iter() {
                Box::pin(self.upsert(vec![record.clone()])).await?;
            }
        } else {
            self.client
                .query(&statement, &params.refs())
                .await
                .map_err(db_err)?;
        }
        Ok(key_texts
            .into_iter()
            .map(|k| read_value(key, Some(k)))
            .collect())
    }

    async fn get(&self, keys: Vec<Value>, include_vectors: bool) -> Result<Vec<Option<Value>>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let key = self.definition.key_field();
        let wanted = self.key_texts(&keys)?;
        let (list, fields) = self.select_list(include_vectors)?;
        let key_index = fields
            .iter()
            .position(|f| f.field_type == FieldType::Key)
            .expect("the key is always selected");
        let statement = format!(
            "SELECT {list} FROM {} WHERE {} = ANY($1::text[]::{}[])",
            self.table,
            quote_identifier(key.effective_storage_name())?,
            base_type(key)?
        );
        let rows = self
            .client
            .query(&statement, &[&wanted])
            .await
            .map_err(db_err)?;
        let mut by_key = std::collections::HashMap::new();
        for row in &rows {
            let k: Option<String> = row.get(key_index);
            by_key.insert(k.unwrap_or_default(), Self::row_record(&fields, row));
        }
        Ok(wanted.iter().map(|k| by_key.get(k).cloned()).collect())
    }

    async fn delete(&self, keys: Vec<Value>) -> Result<()> {
        if keys.is_empty() {
            return Ok(());
        }
        let key = self.definition.key_field();
        let wanted = self.key_texts(&keys)?;
        self.client
            .execute(
                &format!(
                    "DELETE FROM {} WHERE {} = ANY($1::text[]::{}[])",
                    self.table,
                    quote_identifier(key.effective_storage_name())?,
                    base_type(key)?
                ),
                &[&wanted],
            )
            .await
            .map_err(db_err)?;
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
            .ok_or_else(|| config("Select a vector field from the collection definition."))?;
        if field.dimensions.is_some_and(|d| d != vector.len()) {
            return Err(config(format!(
                "query vector has {} dimensions; '{}' has {}",
                vector.len(),
                field.name,
                field.dimensions.unwrap_or_default()
            )));
        }
        let query = vector_text(&serde_json::to_value(&vector)?, &field.name)?;
        let (operator, _) = metric(field)?;
        let mut params = Params::default();
        let vt = cast_type(field)?;
        let column = quote_identifier(field.effective_storage_name())?;
        let q1 = params.push(Some(query.clone()), &vt);
        let distance = |q: &str| format!("({column} {operator} {q})");
        let score = match field.distance_function.as_ref().map(|d| d.as_str()) {
            Some("cosine_similarity") => format!("1 - {}", distance(&q1)),
            Some("dot_prod") => format!("-{}", distance(&q1)),
            _ => distance(&q1),
        };
        let where_sql = self.filter_sql(options, &mut params)?;
        let q2 = params.push(Some(query.clone()), &vt);
        let q3 = params.push(Some(query), &vt);
        let exact = matches!(index_kind(field), "default" | "flat");
        // Adding zero keeps an exact query off the ANN index.
        let ranking = if exact {
            format!("{} + 0", distance(&q3))
        } else {
            distance(&q3)
        };
        let limit = params.push(Some(options.top.to_string()), "bigint");
        let offset = params.push(Some(options.skip.to_string()), "bigint");
        let (list, fields) = self.select_list(options.include_vectors)?;
        let statement = format!(
            "SELECT {list}, ({score})::float8 FROM {} WHERE ({where_sql}) AND {} < 'Infinity'::double precision \
             ORDER BY {ranking} ASC LIMIT {limit} OFFSET {offset}",
            self.table,
            distance(&q2),
        );
        let rows = self
            .client
            .query(&statement, &params.refs())
            .await
            .map_err(db_err)?;
        Ok(rows
            .iter()
            .map(|row| VectorSearchResult {
                record: Self::row_record(&fields, row),
                score: row.get::<_, Option<f64>>(fields.len()),
                score_kind: None,
            })
            .collect())
    }
}

/// A connection to a PostgreSQL database. See the [crate docs](crate).
#[derive(Clone)]
pub struct PostgresStore {
    client: Arc<tokio_postgres::Client>,
    schema: String,
}

impl std::fmt::Debug for PostgresStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PostgresStore")
            .field("schema", &self.schema)
            .finish_non_exhaustive()
    }
}

impl PostgresStore {
    /// Connect with a `tokio-postgres` connection string (key/value or URI),
    /// without TLS, driving the connection on a spawned task.
    pub async fn connect(connection_string: &str) -> Result<Self> {
        if connection_string.trim().is_empty() {
            return Err(config("connection_string must not be empty."));
        }
        let (client, connection) =
            tokio_postgres::connect(connection_string, tokio_postgres::NoTls)
                .await
                .map_err(db_err)?;
        tokio::spawn(async move {
            if let Err(e) = connection.await {
                tracing::warn!(error = %e, "PostgreSQL connection closed with an error");
            }
        });
        Ok(Self::from_client(Arc::new(client)))
    }

    /// Connect using `POSTGRES_CONNECTION_STRING`, as upstream's settings.
    pub async fn from_env() -> Result<Self> {
        let conninfo = std::env::var("POSTGRES_CONNECTION_STRING")
            .map_err(|_| config("POSTGRES_CONNECTION_STRING is not set."))?;
        Self::connect(&conninfo).await
    }

    /// Use a connected client (for example one with TLS).
    pub fn from_client(client: Arc<tokio_postgres::Client>) -> Self {
        Self {
            client,
            schema: "public".into(),
        }
    }

    /// Use this (existing) schema instead of `public`.
    pub fn with_schema(mut self, schema: impl Into<String>) -> Result<Self> {
        let schema = schema.into();
        quote_identifier(&schema)?;
        self.schema = schema;
        Ok(self)
    }

    /// A typed collection handle.
    pub fn collection(
        &self,
        name: &str,
        definition: VectorStoreCollectionDefinition,
    ) -> Result<PostgresCollection> {
        PostgresCollection::new(self.client.clone(), &self.schema, name, definition)
    }
}

#[async_trait]
impl VectorStore for PostgresStore {
    fn get_collection(
        &self,
        name: &str,
        definition: VectorStoreCollectionDefinition,
    ) -> Result<Box<dyn VectorCollection>> {
        Ok(Box::new(self.collection(name, definition)?))
    }

    async fn list_collection_names(&self) -> Result<Vec<String>> {
        let rows = self
            .client
            .query(
                "SELECT table_name::text FROM information_schema.tables WHERE table_schema = $1 \
                 AND table_type = 'BASE TABLE' ORDER BY table_name",
                &[&self.schema],
            )
            .await
            .map_err(db_err)?;
        Ok(rows.iter().map(|r| r.get(0)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::vectors::FilterGroup;

    fn definition() -> VectorStoreCollectionDefinition {
        VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id").with_type("int"),
            VectorStoreField::data("title").with_type("str"),
            VectorStoreField::data("year").with_type("int"),
            VectorStoreField::data("tags").with_type("list"),
            VectorStoreField::vector("embedding", 2),
        ])
        .unwrap()
    }

    fn compile(expression: FilterExpression) -> (String, Vec<Option<String>>) {
        let mut params = Params::default();
        let sql = FilterCompiler {
            definition: &definition(),
            params: &mut params,
        }
        .compile(&expression)
        .unwrap();
        (sql, params.0)
    }

    #[test]
    fn identifiers_are_quoted_and_bounded() {
        assert_eq!(quote_identifier("a\"b").unwrap(), "\"a\"\"b\"");
        assert!(quote_identifier("").is_err());
        assert!(quote_identifier(&"x".repeat(64)).is_err());
    }

    #[test]
    fn filters_compile_to_parameterized_sql() {
        let (sql, params) = compile(Filter::gte("year", 2000).unwrap().into());
        assert_eq!(sql, "(\"year\" >= $1::text::numeric) IS TRUE");
        assert_eq!(params, [Some("2000".to_string())]);
        // A mismatched type is FALSE, not a coercion.
        assert_eq!(
            compile(Filter::eq("year", "2000").unwrap().into()).0,
            "FALSE"
        );
        let (sql, params) = compile(
            Filter::new(
                "title",
                FilterOperator::StartsWith,
                Some(serde_json::json!("50%_off")),
            )
            .unwrap()
            .into(),
        );
        assert_eq!(
            sql,
            "(\"title\" COLLATE \"C\" LIKE $1::text::text ESCAPE '!') IS TRUE"
        );
        assert_eq!(params, [Some("50!%!_off%".to_string())]);
        let group = FilterGroup::not(Filter::is_null("title").unwrap().into()).unwrap();
        assert_eq!(compile(group).0, "(NOT (\"title\" COLLATE \"C\" IS NULL))");
        assert!(FilterCompiler {
            definition: &definition(),
            params: &mut Params::default()
        }
        .compile(&Filter::eq("embedding", 1).unwrap().into())
        .is_err());
    }

    #[test]
    fn values_render_and_read_back() {
        let d = definition();
        let year = d.try_get_field("year").unwrap();
        assert_eq!(
            value_text(year, &serde_json::json!(7)).unwrap().as_deref(),
            Some("7")
        );
        assert!(value_text(year, &serde_json::json!("7")).is_err());
        assert_eq!(read_value(year, Some("7".into())), serde_json::json!(7));
        let v = d.try_get_field("embedding").unwrap();
        assert_eq!(
            value_text(v, &serde_json::json!([1, 0.5]))
                .unwrap()
                .as_deref(),
            Some("[1,0.5]")
        );
        assert_eq!(
            read_value(v, Some("[1,0.5]".into())),
            serde_json::json!([1.0, 0.5])
        );
        assert!(has_timezone("2024-01-01T00:00:00Z"));
        assert!(!has_timezone("2024-01-01T00:00:00"));
    }

    #[test]
    fn untyped_columns_are_refused() {
        assert!(base_type(&VectorStoreField::data("x")).is_err());
        assert_eq!(
            base_type(&VectorStoreField::data("x").with_type("dict")).unwrap(),
            "jsonb"
        );
    }
}
