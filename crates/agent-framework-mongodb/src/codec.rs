//! JSON ⇄ BSON conversion shared by the MongoDB-protocol stores (this crate
//! and `agent-framework-azure-documentdb`).
//!
//! Records in this workspace are JSON. Integers become BSON `Int64`, other
//! numbers `Double` (finite only), and nested arrays and objects convert
//! element by element. Going back, `ObjectId`s read as their 24-character
//! hex string and `DateTime`s as RFC 3339 text, the forms a JSON record
//! carries them in.

use agent_framework_core::error::{Error, Result};
use agent_framework_core::vectors::VectorStoreField;
use mongodb::bson::{Bson, Document};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// The BSON document size limit, 16 MiB.
pub const MAX_DOCUMENT_BYTES: usize = 16 * 1024 * 1024;

/// How strictly object keys inside nested values are checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyRules {
    /// Any key without NUL (MongoDB).
    NoNul,
    /// Nonempty, without NUL or `.`, and not starting with `$`
    /// (Azure DocumentDB).
    TopLevelSafe,
}

/// Convert a JSON value to BSON, refusing non-finite numbers, integers
/// outside the signed 64-bit range, and object keys `rules` forbids.
pub fn json_to_bson(value: &Value, path: &str, rules: KeyRules) -> Result<Bson> {
    Ok(match value {
        Value::Null => Bson::Null,
        Value::Bool(b) => Bson::Boolean(*b),
        Value::String(s) => Bson::String(s.clone()),
        Value::Number(n) => number_to_bson(n, path)?,
        Value::Array(items) => Bson::Array(
            items
                .iter()
                .map(|i| json_to_bson(i, &format!("{path}[]"), rules))
                .collect::<Result<_>>()?,
        ),
        Value::Object(map) => {
            let mut doc = Document::new();
            for (key, item) in map {
                let bad = match rules {
                    KeyRules::NoNul => key.contains('\0'),
                    KeyRules::TopLevelSafe => {
                        key.is_empty()
                            || key.contains('\0')
                            || key.contains('.')
                            || key.starts_with('$')
                    }
                };
                if bad {
                    return Err(Error::Configuration(format!(
                        "Field '{path}' has an object key the store does not accept: {key:?}."
                    )));
                }
                doc.insert(
                    key.clone(),
                    json_to_bson(item, &format!("{path}.{key}"), rules)?,
                );
            }
            Bson::Document(doc)
        }
    })
}

fn number_to_bson(n: &serde_json::Number, path: &str) -> Result<Bson> {
    if let Some(i) = n.as_i64() {
        return Ok(Bson::Int64(i));
    }
    if n.is_u64() {
        return Err(Error::Configuration(format!(
            "Field '{path}' contains an integer outside the signed 64-bit BSON range."
        )));
    }
    match n.as_f64() {
        Some(f) if f.is_finite() => Ok(Bson::Double(f)),
        _ => Err(Error::Configuration(format!(
            "Field '{path}' contains a non-finite number."
        ))),
    }
}

/// Convert BSON back to JSON: `ObjectId` as hex, `DateTime` as RFC 3339,
/// numbers as numbers, anything else in relaxed extended JSON.
pub fn bson_to_json(value: Bson) -> Value {
    match value {
        Bson::Null | Bson::Undefined => Value::Null,
        Bson::Boolean(b) => Value::Bool(b),
        Bson::String(s) => Value::String(s),
        Bson::Int32(i) => Value::from(i),
        Bson::Int64(i) => Value::from(i),
        Bson::Double(f) => serde_json::Number::from_f64(f)
            .map(Value::Number)
            .unwrap_or_else(|| Bson::Double(f).into_relaxed_extjson()),
        Bson::ObjectId(id) => Value::String(id.to_hex()),
        Bson::DateTime(dt) => dt
            .try_to_rfc3339_string()
            .map(Value::String)
            .unwrap_or_else(|_| Bson::DateTime(dt).into_relaxed_extjson()),
        Bson::Array(items) => Value::Array(items.into_iter().map(bson_to_json).collect()),
        Bson::Document(doc) => document_to_json(doc),
        other => other.into_relaxed_extjson(),
    }
}

/// [`bson_to_json`] for a whole document.
pub fn document_to_json(doc: Document) -> Value {
    Value::Object(doc.into_iter().map(|(k, v)| (k, bson_to_json(v))).collect())
}

/// A dense vector: an array of finite numbers (not booleans) of the
/// field's declared dimensions.
pub fn dense_vector(field: &VectorStoreField, value: &Value) -> Result<Vec<f64>> {
    let items = value.as_array().ok_or_else(|| {
        Error::Configuration(format!(
            "Vector field '{}' requires a dense numeric array.",
            field.name
        ))
    })?;
    let numbers = items
        .iter()
        .map(|i| i.as_f64().filter(|f| f.is_finite()))
        .collect::<Option<Vec<f64>>>()
        .ok_or_else(|| {
            Error::Configuration(format!(
                "Vector field '{}' must contain finite numbers.",
                field.name
            ))
        })?;
    if let Some(dims) = field.dimensions {
        if numbers.len() != dims {
            return Err(Error::Configuration(format!(
                "Vector field '{}' expects {dims} dimensions; got {}.",
                field.name,
                numbers.len()
            )));
        }
    }
    Ok(numbers)
}

/// A query vector as BSON doubles, checked like [`dense_vector`].
pub fn query_vector(field: &VectorStoreField, vector: &[f32]) -> Result<Vec<f64>> {
    let as_json = Value::Array(vector.iter().map(|v| serde_json::json!(*v)).collect());
    if vector.iter().any(|v| !v.is_finite()) {
        return Err(Error::Configuration(format!(
            "Vector field '{}' must contain finite numbers.",
            field.name
        )));
    }
    dense_vector(field, &as_json)
}

/// Refuse a document over the 16 MiB BSON limit.
pub fn check_document_size(doc: &Document, what: &str) -> Result<()> {
    let size = mongodb::bson::to_vec(doc)
        .map_err(|e| Error::Configuration(format!("{what} cannot be represented as BSON: {e}")))?
        .len();
    if size > MAX_DOCUMENT_BYTES {
        return Err(Error::Configuration(format!(
            "{what} encodes to {size} bytes and exceeds the 16 MiB BSON document limit."
        )));
    }
    Ok(())
}

/// The lowercase hex SHA-256 of `input`.
pub fn sha256_hex(input: &str) -> String {
    Sha256::digest(input.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// A stable text form of a key, for matching returned documents to the
/// requested keys (`Int32(1)` and `Int64(1)` agree).
pub fn key_string(key: &Bson) -> String {
    match key {
        Bson::Int32(i) => i.to_string(),
        Bson::Int64(i) => i.to_string(),
        Bson::String(s) => format!("s:{s}"),
        Bson::ObjectId(id) => format!("o:{}", id.to_hex()),
        other => other.clone().into_relaxed_extjson().to_string(),
    }
}

/// A JSON object string (relaxed extended JSON) as a filter document, for
/// [`VectorSearchOptions::provider_filter`](agent_framework_core::vectors::VectorSearchOptions::provider_filter).
pub fn provider_filter(raw: &str) -> Result<Document> {
    let value: Value = serde_json::from_str(raw).map_err(|e| {
        Error::Configuration(format!("The provider filter must be a JSON object: {e}"))
    })?;
    match Bson::try_from(value) {
        Ok(Bson::Document(doc)) => Ok(doc),
        Ok(_) => Err(Error::Configuration(
            "The provider filter must be a JSON object.".into(),
        )),
        Err(e) => Err(Error::Configuration(format!(
            "The provider filter is not valid extended JSON: {e}"
        ))),
    }
}
