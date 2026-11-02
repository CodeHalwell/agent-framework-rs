//! A SQL Server [`VectorStore`]: one table per collection in an existing
//! schema, typed columns, and exact dense-vector search over native
//! `VECTOR(n)` columns with `VECTOR_DISTANCE`.
//!
//! Port of upstream `agent_framework_sql_server` (`SqlServerCollection`,
//! `SqlServerStore`) over [`tiberius`] (TDS over rustls). Every operation
//! opens a connection, runs inside one transaction, and closes it, as
//! upstream does.
//!
//! ```no_run
//! use agent_framework_core::vectors::{
//!     Filter, VectorSearchOptions, VectorStore, VectorStoreCollectionDefinition, VectorStoreField,
//! };
//! use agent_framework_sql_server::SqlServerStore;
//!
//! # async fn demo() -> agent_framework_core::error::Result<()> {
//! let store = SqlServerStore::connect(
//!     "server=tcp:localhost,1433;user=sa;password=...;TrustServerCertificate=true",
//! )?;
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
//! - Every field needs a declared type: `str` (`NVARCHAR` with a binary
//!   collation; `NVARCHAR(450)` for keys and indexed fields), `int`
//!   (`BIGINT`), `float` (`FLOAT(53)`), `bool` (`BIT`), `UUID`
//!   (`UNIQUEIDENTIFIER`), `date`, `datetime` (`DATETIME2(7)`, stored in UTC;
//!   a timezone is required), `list` and `dict` (JSON in `NVARCHAR(MAX)`).
//!   Keys are `str`, `int` or `UUID`.
//! - Vectors are `float32` `VECTOR(n)` columns, 1–1998 dimensions. Search is
//!   exact. Distance functions: `DEFAULT` / `cosine_distance`,
//!   `cosine_similarity` (reported as `1 - distance`), `euclidean_distance`,
//!   `dot_prod` (reported as the dot product, higher is closer) and
//!   `negative_dot_prod`.
//! - Data fields marked indexed get a scalar index; tables are created in an
//!   existing schema (default `dbo`) and never migrated.
//! - Portable filters compile to parameterized two-valued T-SQL: string
//!   equality compares bytes (SQL Server ignores trailing spaces otherwise),
//!   ordered operators apply to `int`, `float`, `date` and `datetime`, and
//!   text operators use `LIKE` with escaping. JSON fields are not filterable.
//! - A statement binds at most [`MAX_PARAMETERS`] values; key lookups and
//!   deletes are batched.
//!
//! # Divergences
//!
//! - The `bytes` type is refused: records are JSON, which has no bytes.
//! - Keys are always supplied (the Rust definition has no auto-generated
//!   keys), so there is no `IDENTITY` / `NEWID()` key generation.
//! - There is no `score_threshold` (the Rust search options carry none) and no
//!   filtered listing without a vector (the Rust collection trait has none).
//! - The query timeout bounds a whole operation, not each statement.
//! - Only SQL authentication and the connection-string options `tiberius`
//!   understands are supported (no Windows or Entra authentication).

#![cfg(feature = "experimental-vector-stores")]
#![cfg_attr(docsrs, doc(cfg(feature = "experimental-vector-stores")))]

use std::time::Duration;

use agent_framework_core::error::{Error, Result};
use agent_framework_core::vectors::{
    FieldType, Filter, FilterExpression, FilterGroupOperator, FilterOperator, VectorCollection,
    VectorSearchOptions, VectorSearchResult, VectorStore, VectorStoreCollectionDefinition,
    VectorStoreField,
};
use async_trait::async_trait;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tiberius::{ColumnData, Config, Query, Row};
use tokio::net::TcpStream;
use tokio_util::compat::{Compat, TokioAsyncWriteCompatExt};

/// The most values one statement binds. SQL Server's limit is 2100; this
/// leaves room for paging and search arguments, as upstream.
pub const MAX_PARAMETERS: usize = 2000;
/// Keys per lookup or delete statement.
const KEY_BATCH_SIZE: usize = 1000;
const FLOAT32_MAX: f64 = 3.402_823_466_385_288_6e38;
const COLLATION: &str = "Latin1_General_100_BIN2";
/// The environment variable [`SqlServerStore::from_env`] reads.
pub const CONNECTION_STRING_ENV: &str = "SQL_SERVER_CONNECTION_STRING";

fn config_err(message: impl Into<String>) -> Error {
    Error::Configuration(message.into())
}

fn map_err(e: tiberius::error::Error) -> Error {
    Error::service(format!("SQL Server operation failed: {e}"))
}

fn utf16_len(text: &str) -> usize {
    text.encode_utf16().count()
}

/// Quote one identifier; values are always bound.
pub fn quote_identifier(name: &str) -> Result<String> {
    if name.is_empty() || name.contains('\0') || utf16_len(name) > 128 {
        return Err(config_err(
            "SQL Server identifiers must contain 1-128 UTF-16 code units and no NUL.",
        ));
    }
    Ok(format!("[{}]", name.replace(']', "]]")))
}

/// How a metric's `VECTOR_DISTANCE` result is reported as a score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScoreShape {
    /// The distance itself.
    Distance,
    /// `1 - distance` (cosine similarity).
    Similarity,
    /// `-distance` (`VECTOR_DISTANCE('dot')` is the negative dot product).
    Negative,
}

fn metric(field: &VectorStoreField) -> Result<(&'static str, ScoreShape)> {
    let name = field
        .distance_function
        .as_ref()
        .map(|d| d.as_str())
        .unwrap_or("DEFAULT");
    Ok(match name {
        "DEFAULT" | "cosine_distance" => ("cosine", ScoreShape::Distance),
        "cosine_similarity" => ("cosine", ScoreShape::Similarity),
        "euclidean_distance" => ("euclidean", ScoreShape::Distance),
        "dot_prod" => ("dot", ScoreShape::Negative),
        "negative_dot_prod" => ("dot", ScoreShape::Distance),
        other => {
            return Err(config_err(format!(
                "Unsupported SQL Server distance function '{other}'."
            )))
        }
    })
}

fn is_indexed(field: &VectorStoreField) -> bool {
    field.is_indexed == Some(true)
}

fn column_type(field: &VectorStoreField) -> Result<String> {
    if field.field_type == FieldType::Vector {
        let dims = field
            .dimensions
            .filter(|d| (1..=1998).contains(d))
            .ok_or_else(|| {
                config_err("SQL Server vector dimensions must be an integer between 1 and 1998.")
            })?;
        if !matches!(field.type_.as_deref(), None | Some("float" | "float32")) {
            return Err(config_err(
                "SQL Server VECTOR columns support float32 vectors only.",
            ));
        }
        return Ok(format!("VECTOR({dims})"));
    }
    Ok(match field.type_.as_deref() {
        Some("str") => {
            let width = if field.field_type == FieldType::Key || is_indexed(field) {
                "450"
            } else {
                "MAX"
            };
            format!("NVARCHAR({width}) COLLATE {COLLATION}")
        }
        Some("int") => "BIGINT".into(),
        Some("float") => "FLOAT(53)".into(),
        Some("bool") => "BIT".into(),
        Some("UUID") => "UNIQUEIDENTIFIER".into(),
        Some("date") => "DATE".into(),
        Some("datetime") => "DATETIME2(7)".into(),
        Some("list") | Some("dict") => "NVARCHAR(MAX)".into(),
        Some("bytes") => {
            return Err(config_err(format!(
                "Field '{}': the bytes type is not supported (records are JSON).",
                field.name
            )))
        }
        other => {
            return Err(config_err(format!(
                "Field '{}' needs a supported explicit type; got {other:?}.",
                field.name
            )))
        }
    })
}

/// A bound statement parameter.
#[derive(Debug, Clone, PartialEq)]
pub enum Param {
    /// SQL `NULL` (bound as a null `NVARCHAR`).
    Null,
    /// `NVARCHAR`.
    Text(String),
    /// `BIGINT`.
    BigInt(i64),
    /// `FLOAT`.
    Float(f64),
    /// `BIT`.
    Bit(bool),
}

impl Param {
    fn bind(self, query: &mut Query<'_>) {
        match self {
            Param::Null => query.bind(Option::<String>::None),
            Param::Text(s) => query.bind(s),
            Param::BigInt(i) => query.bind(i),
            Param::Float(f) => query.bind(f),
            Param::Bit(b) => query.bind(b),
        }
    }
}

/// One statement and its parameters, in `@P1…@Pn` order.
#[derive(Debug, Clone, PartialEq)]
pub struct Statement {
    /// The T-SQL text.
    pub sql: String,
    /// The bound values; `@Pi` is `params[i - 1]`.
    pub params: Vec<Param>,
}

impl Statement {
    fn new(sql: impl Into<String>, params: Vec<Param>) -> Result<Self> {
        if params.len() > MAX_PARAMETERS {
            return Err(config_err(format!(
                "SQL Server queries support at most {MAX_PARAMETERS} bound parameters."
            )));
        }
        Ok(Self {
            sql: sql.into(),
            params,
        })
    }
}

/// Parameters collected while building a statement; each push returns its
/// placeholder.
#[derive(Debug, Default)]
struct Params(Vec<Param>);

impl Params {
    fn push(&mut self, param: Param) -> Result<String> {
        if self.0.len() >= MAX_PARAMETERS {
            return Err(config_err(format!(
                "SQL Server queries support at most {MAX_PARAMETERS} bound parameters."
            )));
        }
        self.0.push(param);
        Ok(format!("@P{}", self.0.len()))
    }
}

/// Wrap a placeholder so text-bound values reach the column typed.
fn typed(field: &VectorStoreField, placeholder: &str) -> Result<String> {
    if field.field_type == FieldType::Vector {
        return Ok(format!(
            "CAST({placeholder} AS VECTOR({}))",
            field.dimensions.unwrap_or_default()
        ));
    }
    Ok(match field.type_.as_deref() {
        Some("UUID") => format!("CAST({placeholder} AS UNIQUEIDENTIFIER)"),
        Some("date") => format!("CAST({placeholder} AS DATE)"),
        Some("datetime") => format!(
            "CONVERT(DATETIME2(7), SWITCHOFFSET(CAST({placeholder} AS DATETIMEOFFSET(7)), '+00:00'))"
        ),
        _ => placeholder.to_string(),
    })
}

fn json_check(value: &Value) -> Result<()> {
    match value {
        Value::Number(n) if !n.as_f64().is_some_and(f64::is_finite) => {
            Err(config_err("JSON values must be finite."))
        }
        Value::Array(items) => items.iter().try_for_each(json_check),
        Value::Object(map) => map.values().try_for_each(json_check),
        _ => Ok(()),
    }
}

fn vector_param(field: &VectorStoreField, value: &Value) -> Result<Param> {
    let items = value.as_array().ok_or_else(|| {
        config_err(format!(
            "Vector field '{}' requires a dense numeric sequence.",
            field.name
        ))
    })?;
    let dims = field.dimensions.unwrap_or_default();
    if items.len() != dims {
        return Err(config_err(format!(
            "Vector field '{}' requires {dims} dimensions.",
            field.name
        )));
    }
    let numbers = items
        .iter()
        .map(|i| {
            i.as_f64()
                .filter(|f| f.is_finite() && f.abs() <= FLOAT32_MAX)
        })
        .collect::<Option<Vec<f64>>>()
        .ok_or_else(|| {
            config_err(format!(
                "Vector field '{}' requires finite float32 elements.",
                field.name
            ))
        })?;
    Ok(Param::Text(serde_json::to_string(&numbers)?))
}

fn has_timezone(text: &str) -> bool {
    let Some(t) = text.find(['T', 't', ' ']) else {
        return false;
    };
    let time = &text[t + 1..];
    time.ends_with(['Z', 'z'])
        || time
            .rfind(['+', '-'])
            .is_some_and(|i| time[i..].contains(':'))
}

/// A key or data value as a parameter, checked against the declared type.
fn value_param(field: &VectorStoreField, value: &Value) -> Result<Param> {
    if value.is_null() {
        if field.field_type == FieldType::Key {
            return Err(config_err("SQL Server keys cannot be null."));
        }
        return Ok(Param::Null);
    }
    if field.field_type == FieldType::Vector {
        return vector_param(field, value);
    }
    let kind = field.type_.as_deref().unwrap_or_default();
    let mismatch = || {
        config_err(format!(
            "Field '{}' requires a value of type '{kind}'.",
            field.name
        ))
    };
    Ok(match (kind, value) {
        ("str", Value::String(s)) => {
            if (field.field_type == FieldType::Key || is_indexed(field)) && utf16_len(s) > 450 {
                return Err(config_err(format!(
                    "Field '{}' exceeds the indexed NVARCHAR(450) limit.",
                    field.name
                )));
            }
            if field.field_type == FieldType::Key && s.ends_with(' ') {
                return Err(config_err(
                    "SQL Server string keys cannot end in a space; SQL Server ignores trailing spaces in keys.",
                ));
            }
            Param::Text(s.clone())
        }
        ("int", Value::Number(n)) => Param::BigInt(n.as_i64().ok_or_else(|| {
            config_err(format!(
                "Field '{}' requires an integer in the SQL Server bigint range.",
                field.name
            ))
        })?),
        ("float", Value::Number(n)) => {
            Param::Float(n.as_f64().filter(|f| f.is_finite()).ok_or_else(|| {
                config_err(format!("Field '{}' requires a finite number.", field.name))
            })?)
        }
        ("bool", Value::Bool(b)) => Param::Bit(*b),
        ("UUID", Value::String(s)) => Param::Text(
            uuid::Uuid::parse_str(s)
                .map_err(|_| mismatch())?
                .hyphenated()
                .to_string(),
        ),
        ("date", Value::String(s)) => Param::Text(s.clone()),
        ("datetime", Value::String(s)) => {
            if !has_timezone(s) {
                return Err(config_err(format!(
                    "Datetime field '{}' requires a timezone.",
                    field.name
                )));
            }
            Param::Text(s.clone())
        }
        ("list", Value::Array(_)) | ("dict", Value::Object(_)) => {
            json_check(value)?;
            Param::Text(value.to_string())
        }
        _ => return Err(mismatch()),
    })
}

/// A numeric filter operand: integral values bind as `BIGINT`, others as
/// `FLOAT`. A float column accepts integral operands outside the bigint
/// range; an int column refuses them.
fn numeric_operand(field: &VectorStoreField, n: &serde_json::Number) -> Result<Param> {
    if let Some(i) = n.as_i64() {
        return Ok(Param::BigInt(i));
    }
    let f = n
        .as_f64()
        .filter(|f| f.is_finite())
        .ok_or_else(|| config_err("Numeric filter values must be finite."))?;
    if field.type_.as_deref() == Some("int") && f.fract() == 0.0 {
        return Err(config_err(
            "Integer filter values must fit in the SQL Server bigint range.",
        ));
    }
    Ok(Param::Float(f))
}

/// Compiles portable filters into a parameterized two-valued predicate.
struct FilterCompiler<'a> {
    definition: &'a VectorStoreCollectionDefinition,
    alias: &'a str,
    params: &'a mut Params,
}

impl FilterCompiler<'_> {
    fn compile(&mut self, expression: Option<&FilterExpression>) -> Result<String> {
        match expression {
            None => Ok("1 = 1".into()),
            Some(e) => self.condition(e),
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
        let kind = field.type_.as_deref().unwrap_or_default();
        if matches!(kind, "list" | "dict") {
            return Err(config_err(
                "Equality filtering JSON fields is not supported.",
            ));
        }
        let param = match (kind, value) {
            ("int" | "float", Value::Number(n)) => numeric_operand(field, n)?,
            ("str", Value::String(_))
            | ("bool", Value::Bool(_))
            | ("UUID" | "date" | "datetime", Value::String(_)) => value_param(field, value)?,
            // A value of another type never equals the column.
            _ => return Ok("1 = 0".into()),
        };
        let ph = self.params.push(param)?;
        if kind == "str" {
            // SQL Server pads trailing spaces even with binary collations;
            // compare bytes instead.
            return Ok(format!(
                "({column} IS NOT NULL AND CONVERT(VARBINARY(MAX), {column}) = \
                 CONVERT(VARBINARY(MAX), CONVERT(NVARCHAR(MAX), {ph})))"
            ));
        }
        Ok(format!(
            "({column} IS NOT NULL AND {column} = {})",
            typed(field, &ph)?
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
            return Err(config_err(
                "SQL Server filters and ordering do not support nested field paths.",
            ));
        }
        let field = self
            .definition
            .try_get_field(&filter.field_name)
            .ok_or_else(|| {
                config_err(format!("Unknown SQL Server field '{}'.", filter.field_name))
            })?;
        if field.field_type == FieldType::Vector {
            return Err(config_err(
                "Filtering and ordering vector columns is not supported.",
            ));
        }
        let column = format!(
            "{}{}",
            self.alias,
            quote_identifier(field.effective_storage_name())?
        );
        let value = filter.value.clone().unwrap_or(Value::Null);
        Ok(match &filter.operator {
            // SQL columns exist even when their values are NULL.
            FilterOperator::Exists => "1 = 1".into(),
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
                    "1 = 0".to_string()
                } else {
                    parts.join(" OR ")
                };
                let membership = if *op == FilterOperator::In {
                    format!("({choices})")
                } else {
                    format!("(NOT ({choices}))")
                };
                format!("({column} IS NOT NULL AND {membership})")
            }
            op @ (FilterOperator::Gt
            | FilterOperator::Gte
            | FilterOperator::Lt
            | FilterOperator::Lte
            | FilterOperator::Between) => {
                let kind = field.type_.as_deref().unwrap_or_default();
                if !matches!(kind, "int" | "float" | "date" | "datetime") {
                    return Err(config_err(format!(
                        "Ordered SQL Server filtering is not supported for '{kind}'."
                    )));
                }
                let operands = if *op == FilterOperator::Between {
                    value.as_array().cloned().unwrap_or_default()
                } else {
                    vec![value.clone()]
                };
                let mut placeholders = Vec::new();
                for operand in &operands {
                    let param = match (kind, operand) {
                        (_, Value::Null | Value::Bool(_)) => {
                            return Err(config_err(
                                "Ordered filter operands must be non-null scalars of the column's type.",
                            ))
                        }
                        ("int" | "float", Value::Number(n)) => numeric_operand(field, n)?,
                        _ => value_param(field, operand)?,
                    };
                    let ph = self.params.push(param)?;
                    placeholders.push(typed(field, &ph)?);
                }
                let comparison = match op {
                    FilterOperator::Between => {
                        if placeholders.len() != 2 {
                            return Err(config_err("between requires two operands."));
                        }
                        format!(
                            "{column} BETWEEN {} AND {}",
                            placeholders[0], placeholders[1]
                        )
                    }
                    FilterOperator::Gt => format!("{column} > {}", placeholders[0]),
                    FilterOperator::Gte => format!("{column} >= {}", placeholders[0]),
                    FilterOperator::Lt => format!("{column} < {}", placeholders[0]),
                    _ => format!("{column} <= {}", placeholders[0]),
                };
                format!("({column} IS NOT NULL AND {comparison})")
            }
            op @ (FilterOperator::ContainsText
            | FilterOperator::StartsWith
            | FilterOperator::EndsWith) => {
                if field.type_.as_deref() != Some("str") {
                    return Err(config_err("Text filtering requires a string column."));
                }
                let text = value
                    .as_str()
                    .ok_or_else(|| config_err("Text filtering requires a string operand."))?;
                if text.contains('\0') {
                    return Err(config_err(
                        "SQL Server LIKE does not support NUL characters.",
                    ));
                }
                let escaped = text
                    .replace('!', "!!")
                    .replace('%', "!%")
                    .replace('_', "!_")
                    .replace('[', "[[]");
                let pattern = match op {
                    FilterOperator::StartsWith => format!("{escaped}%"),
                    FilterOperator::EndsWith => format!("%{escaped}"),
                    _ => format!("%{escaped}%"),
                };
                if utf16_len(&pattern) * 2 > 8000 {
                    return Err(config_err(
                        "SQL Server LIKE patterns cannot exceed 8000 bytes.",
                    ));
                }
                let ph = self.params.push(Param::Text(pattern))?;
                format!(
                    "({column} IS NOT NULL AND {column} COLLATE {COLLATION} LIKE {ph} ESCAPE '!')"
                )
            }
            other => {
                return Err(config_err(format!(
                    "Unsupported SQL Server filter operator '{other:?}'."
                )))
            }
        })
    }
}

/// Opens a connection per operation and runs its statements in one
/// transaction.
#[derive(Clone)]
struct Client {
    config: Config,
    timeout: Option<Duration>,
}

type Connection = tiberius::Client<Compat<TcpStream>>;

impl Client {
    async fn connect(&self) -> Result<Connection> {
        let tcp = TcpStream::connect(self.config.get_addr())
            .await
            .map_err(|e| Error::service(format!("SQL Server connection failed: {e}")))?;
        tcp.set_nodelay(true)
            .map_err(|e| Error::service(format!("SQL Server connection failed: {e}")))?;
        tiberius::Client::connect(self.config.clone(), tcp.compat_write())
            .await
            .map_err(map_err)
    }

    /// Run `statements` in one transaction; the rows of each, in order.
    async fn run(&self, statements: Vec<Statement>) -> Result<Vec<Vec<Row>>> {
        let work = async move {
            let mut connection = self.connect().await?;
            let outcome = Self::transaction(&mut connection, statements).await;
            if outcome.is_err() {
                let _ = connection
                    .simple_query("IF @@TRANCOUNT > 0 ROLLBACK TRANSACTION")
                    .await;
            }
            let closed = connection.close().await;
            match (outcome, closed) {
                (Ok(rows), Ok(())) => Ok(rows),
                (Ok(_), Err(e)) => Err(Error::service(format!(
                    "SQL Server transaction committed, but connection cleanup failed; \
                     do not retry this operation automatically: {e}"
                ))),
                (Err(e), _) => Err(e),
            }
        };
        match self.timeout {
            Some(limit) if !limit.is_zero() => tokio::time::timeout(limit, work)
                .await
                .map_err(|_| Error::service("SQL Server operation timed out."))?,
            _ => work.await,
        }
    }

    async fn transaction(
        connection: &mut Connection,
        statements: Vec<Statement>,
    ) -> Result<Vec<Vec<Row>>> {
        connection
            .simple_query("SET XACT_ABORT ON; BEGIN TRANSACTION")
            .await
            .map_err(map_err)?
            .into_results()
            .await
            .map_err(map_err)?;
        let mut results = Vec::with_capacity(statements.len());
        for statement in statements {
            let mut query = Query::new(statement.sql);
            for param in statement.params {
                param.bind(&mut query);
            }
            let rows = query
                .query(connection)
                .await
                .map_err(map_err)?
                .into_first_result()
                .await
                .map_err(map_err)?;
            results.push(rows);
        }
        connection
            .simple_query("COMMIT TRANSACTION")
            .await
            .map_err(map_err)?
            .into_results()
            .await
            .map_err(map_err)?;
        Ok(results)
    }
}

/// A cell as JSON. The projection converts dates, UUIDs and vectors to text,
/// so only scalar wire types arrive here.
fn cell(data: ColumnData<'static>) -> Result<Value> {
    Ok(match data {
        ColumnData::U8(v) => v.map(Value::from).unwrap_or(Value::Null),
        ColumnData::I16(v) => v.map(Value::from).unwrap_or(Value::Null),
        ColumnData::I32(v) => v.map(Value::from).unwrap_or(Value::Null),
        ColumnData::I64(v) => v.map(Value::from).unwrap_or(Value::Null),
        ColumnData::F32(v) => v
            .map(|f| serde_json::json!(f64::from(f)))
            .unwrap_or(Value::Null),
        ColumnData::F64(v) => v.map(|f| serde_json::json!(f)).unwrap_or(Value::Null),
        ColumnData::Bit(v) => v.map(Value::Bool).unwrap_or(Value::Null),
        ColumnData::String(v) => v
            .map(|s| Value::String(s.into_owned()))
            .unwrap_or(Value::Null),
        ColumnData::Guid(v) => v
            .map(|g| Value::String(g.to_string().to_ascii_lowercase()))
            .unwrap_or(Value::Null),
        other => {
            return Err(Error::service(format!(
                "SQL Server returned an unexpected column type {other:?}."
            )))
        }
    })
}

/// Parse a stored cell into the field's JSON shape.
fn parse_cell(field: &VectorStoreField, value: Value) -> Result<Value> {
    if value.is_null() {
        return Ok(value);
    }
    let bad =
        |what: &str| Error::service(format!("SQL Server returned {what} for '{}'.", field.name));
    let kind = field.type_.as_deref().unwrap_or_default();
    if field.field_type == FieldType::Vector || matches!(kind, "list" | "dict") {
        let text = value.as_str().ok_or_else(|| bad("a non-JSON value"))?;
        let parsed: Value = serde_json::from_str(text).map_err(|_| bad("invalid JSON"))?;
        if field.field_type == FieldType::Vector {
            let items = parsed.as_array().ok_or_else(|| bad("an invalid vector"))?;
            if items.len() != field.dimensions.unwrap_or_default()
                || !items.iter().all(|i| {
                    i.as_f64()
                        .is_some_and(|f| f.is_finite() && f.abs() <= FLOAT32_MAX)
                })
            {
                return Err(bad("an invalid vector"));
            }
            return Ok(parsed);
        }
        let right = if kind == "list" {
            parsed.is_array()
        } else {
            parsed.is_object()
        };
        if !right {
            return Err(bad("the wrong JSON type"));
        }
        return Ok(parsed);
    }
    if kind == "bool" {
        if let Some(n) = value.as_i64() {
            return match n {
                0 => Ok(Value::Bool(false)),
                1 => Ok(Value::Bool(true)),
                _ => Err(bad("an invalid boolean")),
            };
        }
    }
    Ok(value)
}

/// A table in a SQL Server schema. See the [crate docs](crate).
#[derive(Clone)]
pub struct SqlServerCollection {
    client: Client,
    name: String,
    schema: String,
    table: String,
    definition: VectorStoreCollectionDefinition,
}

impl std::fmt::Debug for SqlServerCollection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqlServerCollection")
            .field("schema", &self.schema)
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl SqlServerCollection {
    fn new(
        client: Client,
        schema: &str,
        name: &str,
        definition: VectorStoreCollectionDefinition,
    ) -> Result<Self> {
        let table = format!("{}.{}", quote_identifier(schema)?, quote_identifier(name)?);
        for field in definition.fields() {
            quote_identifier(field.effective_storage_name())?;
            column_type(field)?;
            if field.is_full_text_indexed == Some(true) {
                return Err(config_err(
                    "SQL Server full-text and keyword-hybrid search are not supported.",
                ));
            }
            match field.field_type {
                FieldType::Vector => {
                    metric(field)?;
                    match field.index_kind.as_ref().map(|k| k.as_str()) {
                        None | Some("default" | "flat") => {}
                        Some(_) => {
                            return Err(config_err(
                                "SQL Server approximate vector indexes are not supported.",
                            ))
                        }
                    }
                }
                FieldType::Key => {
                    if !matches!(field.type_.as_deref(), Some("str" | "int" | "UUID")) {
                        return Err(config_err("SQL Server keys must be str, int or UUID."));
                    }
                }
                FieldType::Data => {
                    if is_indexed(field) && matches!(field.type_.as_deref(), Some("list" | "dict"))
                    {
                        return Err(config_err(format!(
                            "SQL Server cannot index data fields of type '{}'.",
                            field.type_.as_deref().unwrap_or_default()
                        )));
                    }
                }
            }
        }
        Ok(Self {
            client,
            name: name.to_string(),
            schema: schema.to_string(),
            table,
            definition,
        })
    }

    /// The schema holding the table.
    pub fn schema(&self) -> &str {
        &self.schema
    }

    fn index_name(&self, column: &str) -> String {
        let digest = Sha256::digest(format!("{}\0{}\0{column}", self.schema, self.name));
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        format!("af_sql_{}", &hex[..32])
    }

    /// The statements [`VectorCollection::ensure_collection_exists`] runs:
    /// a guarded `CREATE TABLE`, then a guarded `CREATE INDEX` per indexed
    /// data field (test seam).
    pub fn create_statements(&self) -> Result<Vec<Statement>> {
        let columns = self
            .definition
            .fields()
            .iter()
            .map(|f| {
                let column = quote_identifier(f.effective_storage_name())?;
                let suffix = if f.field_type == FieldType::Key {
                    " NOT NULL PRIMARY KEY"
                } else {
                    " NULL"
                };
                Ok(format!("{column} {}{suffix}", column_type(f)?))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut statements = vec![Statement::new(
            format!(
                "IF NOT EXISTS (SELECT 1 FROM sys.tables AS t \
                 JOIN sys.schemas AS s ON s.schema_id = t.schema_id \
                 WHERE s.name = @P1 AND t.name = @P2) \
                 BEGIN CREATE TABLE {} ({}) END",
                self.table,
                columns.join(", ")
            ),
            vec![
                Param::Text(self.schema.clone()),
                Param::Text(self.name.clone()),
            ],
        )?];
        for field in self.definition.fields() {
            if field.field_type != FieldType::Data || !is_indexed(field) {
                continue;
            }
            let column = field.effective_storage_name();
            let index = self.index_name(column);
            statements.push(Statement::new(
                format!(
                    "IF NOT EXISTS (SELECT 1 FROM sys.indexes AS i \
                     JOIN sys.tables AS t ON t.object_id = i.object_id \
                     JOIN sys.schemas AS s ON s.schema_id = t.schema_id \
                     WHERE s.name = @P1 AND t.name = @P2 AND i.name = @P3) \
                     CREATE INDEX {} ON {} ({})",
                    quote_identifier(&index)?,
                    self.table,
                    quote_identifier(column)?
                ),
                vec![
                    Param::Text(self.schema.clone()),
                    Param::Text(self.name.clone()),
                    Param::Text(index),
                ],
            )?);
        }
        Ok(statements)
    }

    /// The selected fields, and a projection reading each as a scalar wire
    /// type (dates, UUIDs and vectors as text).
    fn projection(&self, include_vectors: bool) -> Result<(Vec<VectorStoreField>, String)> {
        let mut fields = Vec::new();
        let mut columns = Vec::new();
        for field in self.definition.fields() {
            if !include_vectors && field.field_type == FieldType::Vector {
                continue;
            }
            let column = format!("t.{}", quote_identifier(field.effective_storage_name())?);
            columns.push(if field.field_type == FieldType::Vector {
                format!("CAST({column} AS NVARCHAR(MAX))")
            } else {
                match field.type_.as_deref() {
                    Some("UUID") => format!("LOWER(CONVERT(NVARCHAR(36), {column}))"),
                    Some("date") => format!("CONVERT(NVARCHAR(10), {column}, 23)"),
                    Some("datetime") => format!("CONVERT(NVARCHAR(40), {column}, 127) + N'Z'"),
                    _ => column,
                }
            });
            fields.push(field.clone());
        }
        Ok((fields, columns.join(", ")))
    }

    fn record(
        &self,
        fields: &[VectorStoreField],
        row: Row,
        include_vectors: bool,
    ) -> Result<(Value, Vec<Value>)> {
        let cells = row.into_iter().map(cell).collect::<Result<Vec<_>>>()?;
        if cells.len() < fields.len() {
            return Err(Error::service(
                "SQL Server returned a row with an unexpected column count.",
            ));
        }
        let mut cells = cells.into_iter();
        let mut stored = serde_json::Map::new();
        for field in fields {
            let value = parse_cell(field, cells.next().unwrap_or(Value::Null))?;
            if !value.is_null() {
                stored.insert(field.effective_storage_name().to_string(), value);
            }
        }
        let record = self
            .definition
            .from_storage(&Value::Object(stored), include_vectors)?;
        Ok((record, cells.collect()))
    }

    /// The statements [`VectorCollection::upsert`] runs, one per record:
    /// a locked existence check, then `UPDATE` or `INSERT` (test seam).
    pub fn upsert_statements(&self, records: &[Value]) -> Result<(Vec<Statement>, Vec<Value>)> {
        let key_name = self.definition.key_field_storage_name().to_string();
        let key_column = quote_identifier(&key_name)?;
        let fields = self.definition.fields();
        let mut statements = Vec::with_capacity(records.len());
        let mut keys = Vec::with_capacity(records.len());
        for record in records {
            let stored = self.definition.to_storage(record)?;
            let key = stored.get(&key_name).cloned().ok_or_else(|| {
                config_err(format!(
                    "Record is missing vector store field '{}'.",
                    self.definition.key_field().name
                ))
            })?;
            let mut params = Params::default();
            let mut names = Vec::new();
            let mut values = Vec::new();
            let mut key_ph = String::new();
            for field in fields {
                let column = quote_identifier(field.effective_storage_name())?;
                let value = stored
                    .get(field.effective_storage_name())
                    .unwrap_or(&Value::Null);
                let ph = typed(field, &params.push(value_param(field, value)?)?)?;
                if field.field_type == FieldType::Key {
                    key_ph = ph.clone();
                }
                names.push(column);
                values.push(ph);
            }
            let assignments: Vec<String> = names
                .iter()
                .zip(&values)
                .filter(|(n, _)| **n != key_column)
                .map(|(n, v)| format!("{n} = {v}"))
                .collect();
            let assignments = if assignments.is_empty() {
                format!("{key_column} = {key_column}")
            } else {
                assignments.join(", ")
            };
            let table = &self.table;
            statements.push(Statement::new(
                format!(
                    "IF EXISTS (SELECT 1 FROM {table} WITH (UPDLOCK, HOLDLOCK) WHERE {key_column} = {key_ph}) \
                     UPDATE {table} SET {assignments} WHERE {key_column} = {key_ph} \
                     ELSE INSERT INTO {table} ({}) VALUES ({})",
                    names.join(", "),
                    values.join(", ")
                ),
                params.0,
            )?);
            keys.push(key);
        }
        Ok((statements, keys))
    }

    fn key_batches(&self, keys: &[Value], head: &str) -> Result<Vec<Statement>> {
        let key = self.definition.key_field();
        let column = format!(
            "t.{}",
            quote_identifier(self.definition.key_field_storage_name())?
        );
        keys.chunks(KEY_BATCH_SIZE)
            .map(|batch| {
                let mut params = Params::default();
                let markers = batch
                    .iter()
                    .map(|k| typed(key, &params.push(value_param(key, k)?)?))
                    .collect::<Result<Vec<_>>>()?;
                Statement::new(
                    format!("{head} WHERE {column} IN ({})", markers.join(", ")),
                    params.0,
                )
            })
            .collect()
    }

    /// The search statement and its parameters (test seam).
    pub fn search_sql(&self, vector: &[f32], options: &VectorSearchOptions) -> Result<Statement> {
        Ok(self.search_statement(vector, options)?.0)
    }

    fn search_statement(
        &self,
        vector: &[f32],
        options: &VectorSearchOptions,
    ) -> Result<(Statement, Vec<VectorStoreField>, ScoreShape)> {
        let field = self
            .definition
            .try_get_vector_field(options.vector_field_name.as_deref())
            .ok_or_else(|| config_err("Select a vector field from the collection definition."))?;
        let (metric, shape) = metric(field)?;
        let mut params = Params::default();
        let query = params.push(vector_param(
            field,
            &Value::Array(vector.iter().map(|v| serde_json::json!(*v)).collect()),
        )?)?;
        let mut predicate = FilterCompiler {
            definition: &self.definition,
            alias: "t.",
            params: &mut params,
        }
        .compile(options.filter.as_ref())?;
        if let Some(raw) = options
            .provider_filter
            .as_deref()
            .filter(|r| !r.trim().is_empty())
        {
            predicate = format!("({predicate}) AND ({raw})");
        }
        let (fields, projection) = self.projection(options.include_vectors)?;
        let vector_column = format!("t.{}", quote_identifier(field.effective_storage_name())?);
        let key_column = format!(
            "t.{}",
            quote_identifier(self.definition.key_field_storage_name())?
        );
        let skip = params.push(Param::BigInt(options.skip as i64))?;
        let top = params.push(Param::BigInt(options.top as i64))?;
        let sql = format!(
            "SELECT {projection}, d.[distance] FROM {} AS t \
             CROSS APPLY (VALUES (VECTOR_DISTANCE('{metric}', CAST({query} AS VECTOR({})), {vector_column}))) AS d([distance]) \
             WHERE {vector_column} IS NOT NULL AND ({predicate}) AND d.[distance] IS NOT NULL \
             ORDER BY d.[distance] ASC, {key_column} ASC OFFSET {skip} ROWS FETCH NEXT {top} ROWS ONLY",
            self.table,
            field.dimensions.unwrap_or_default(),
        );
        Ok((Statement::new(sql, params.0)?, fields, shape))
    }
}

/// A key compared as text, so UUIDs match whatever their case.
fn normalized_key(value: &Value) -> String {
    match value {
        Value::String(s) => uuid::Uuid::parse_str(s)
            .map(|u| u.hyphenated().to_string())
            .unwrap_or_else(|_| s.clone()),
        other => other.to_string(),
    }
}

#[async_trait]
impl VectorCollection for SqlServerCollection {
    fn name(&self) -> &str {
        &self.name
    }

    fn definition(&self) -> &VectorStoreCollectionDefinition {
        &self.definition
    }

    async fn ensure_collection_exists(&self) -> Result<()> {
        self.client.run(self.create_statements()?).await.map(|_| ())
    }

    async fn collection_exists(&self) -> Result<bool> {
        let rows = self
            .client
            .run(vec![Statement::new(
                "SELECT 1 FROM sys.tables AS t JOIN sys.schemas AS s ON s.schema_id = t.schema_id \
                 WHERE s.name = @P1 AND t.name = @P2",
                vec![
                    Param::Text(self.schema.clone()),
                    Param::Text(self.name.clone()),
                ],
            )?])
            .await?;
        Ok(rows.first().is_some_and(|r| !r.is_empty()))
    }

    async fn ensure_collection_deleted(&self) -> Result<()> {
        self.client
            .run(vec![Statement::new(
                format!("DROP TABLE IF EXISTS {}", self.table),
                Vec::new(),
            )?])
            .await
            .map(|_| ())
    }

    async fn upsert(&self, records: Vec<Value>) -> Result<Vec<Value>> {
        if records.is_empty() {
            return Ok(Vec::new());
        }
        let (statements, keys) = self.upsert_statements(&records)?;
        self.client.run(statements).await?;
        Ok(keys)
    }

    async fn get(&self, keys: Vec<Value>, include_vectors: bool) -> Result<Vec<Option<Value>>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let (fields, projection) = self.projection(include_vectors)?;
        let statements = self.key_batches(
            &keys,
            &format!("SELECT {projection} FROM {} AS t", self.table),
        )?;
        let results = self.client.run(statements).await?;
        let key_name = self.definition.key_field().name.clone();
        let mut found = std::collections::HashMap::new();
        for row in results.into_iter().flatten() {
            let (record, _) = self.record(&fields, row, include_vectors)?;
            let id = record
                .get(&key_name)
                .map(normalized_key)
                .unwrap_or_default();
            found.insert(id, record);
        }
        Ok(keys
            .iter()
            .map(|k| found.get(&normalized_key(k)).cloned())
            .collect())
    }

    async fn delete(&self, keys: Vec<Value>) -> Result<()> {
        if keys.is_empty() {
            return Ok(());
        }
        let statements = self.key_batches(&keys, &format!("DELETE t FROM {} AS t", self.table))?;
        self.client.run(statements).await.map(|_| ())
    }

    async fn search(
        &self,
        vector: Vec<f32>,
        options: &VectorSearchOptions,
    ) -> Result<Vec<VectorSearchResult>> {
        options.validate()?;
        let (statement, fields, shape) = self.search_statement(&vector, options)?;
        let rows = self.client.run(vec![statement]).await?;
        rows.into_iter()
            .flatten()
            .map(|row| {
                let (record, rest) = self.record(&fields, row, options.include_vectors)?;
                let distance = rest
                    .first()
                    .and_then(Value::as_f64)
                    .filter(|d| d.is_finite())
                    .ok_or_else(|| {
                        Error::service("SQL Server returned a non-finite vector distance.")
                    })?;
                let score = match shape {
                    ScoreShape::Distance => distance,
                    ScoreShape::Similarity => 1.0 - distance,
                    ScoreShape::Negative => -distance,
                };
                Ok(VectorSearchResult {
                    record,
                    score: Some(score),
                    score_kind: None,
                })
            })
            .collect()
    }
}

/// A SQL Server database schema holding vector collections. See the
/// [crate docs](crate).
#[derive(Clone)]
pub struct SqlServerStore {
    client: Client,
    schema: String,
}

impl std::fmt::Debug for SqlServerStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqlServerStore")
            .field("schema", &self.schema)
            .finish_non_exhaustive()
    }
}

impl SqlServerStore {
    /// From an ADO.NET connection string (`server=tcp:host,1433;user=…;
    /// password=…;database=…;TrustServerCertificate=true`). No I/O: each
    /// operation connects.
    pub fn connect(connection_string: &str) -> Result<Self> {
        if connection_string.trim().is_empty() {
            return Err(config_err("connection_string must not be empty."));
        }
        let config = Config::from_ado_string(connection_string)
            .map_err(|e| config_err(format!("Invalid SQL Server connection string: {e}")))?;
        Ok(Self::from_config(config))
    }

    /// From [`CONNECTION_STRING_ENV`] (`SQL_SERVER_CONNECTION_STRING`), as
    /// upstream.
    pub fn from_env() -> Result<Self> {
        let value = std::env::var(CONNECTION_STRING_ENV)
            .ok()
            .filter(|v| !v.trim().is_empty())
            .ok_or_else(|| {
                config_err(
                    "SQL_SERVER_CONNECTION_STRING or an explicit connection_string is required.",
                )
            })?;
        Self::connect(&value)
    }

    /// From a prepared `tiberius` configuration.
    pub fn from_config(config: Config) -> Self {
        Self {
            client: Client {
                config,
                timeout: None,
            },
            schema: "dbo".into(),
        }
    }

    /// Use an existing schema other than `dbo`.
    pub fn with_schema(mut self, schema: impl Into<String>) -> Result<Self> {
        let schema = schema.into();
        quote_identifier(&schema)?;
        self.schema = schema;
        Ok(self)
    }

    /// Bound each operation (connect, transaction, close); zero disables it.
    pub fn with_query_timeout(mut self, timeout: Duration) -> Self {
        self.client.timeout = Some(timeout);
        self
    }

    /// The schema collections live in.
    pub fn schema(&self) -> &str {
        &self.schema
    }

    /// A typed collection handle (no I/O).
    pub fn collection(
        &self,
        name: &str,
        definition: VectorStoreCollectionDefinition,
    ) -> Result<SqlServerCollection> {
        SqlServerCollection::new(self.client.clone(), &self.schema, name, definition)
    }

    /// Drop a collection's table in this schema, if it exists.
    pub async fn ensure_collection_deleted(&self, name: &str) -> Result<()> {
        let table = format!(
            "{}.{}",
            quote_identifier(&self.schema)?,
            quote_identifier(name)?
        );
        self.client
            .run(vec![Statement::new(
                format!("DROP TABLE IF EXISTS {table}"),
                Vec::new(),
            )?])
            .await
            .map(|_| ())
    }
}

#[async_trait]
impl VectorStore for SqlServerStore {
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
            .run(vec![Statement::new(
                "SELECT t.name FROM sys.tables AS t JOIN sys.schemas AS s ON s.schema_id = t.schema_id \
                 WHERE s.name = @P1 ORDER BY t.name",
                vec![Param::Text(self.schema.clone())],
            )?])
            .await?;
        rows.into_iter()
            .flatten()
            .map(|row| {
                row.into_iter()
                    .next()
                    .map(cell)
                    .transpose()?
                    .and_then(|v| v.as_str().map(str::to_string))
                    .ok_or_else(|| Error::service("SQL Server returned a non-string table name."))
            })
            .collect()
    }
}
