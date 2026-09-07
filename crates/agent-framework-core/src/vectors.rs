//! Vector-store abstractions: describe a collection's fields, then read and
//! write records through a provider-agnostic trait pair.
//!
//! Ports the portable core of upstream's `agent_framework._vectors` (#8014):
//! [`VectorStoreField`] and [`VectorStoreCollectionDefinition`] describe a
//! collection's shape, [`VectorCollection`] is one collection's data plane,
//! and [`VectorStore`] is the connection that hands out collections. A
//! provider crate implements the two traits; nothing here talks to a service.
//!
//! # Divergence: records are `serde_json::Value`, not a registered model type
//!
//! Roughly half of upstream's module is a model layer: a
//! `@vector_store_model` decorator that registers a Python class, walks its
//! annotations, and generates encoders/decoders between instances of that
//! class and the flat mappings a store wants. Rust needs none of it — a
//! record type derives `Serialize`/`Deserialize` and `serde` already performs
//! that conversion, with the field names checked at compile time rather than
//! at registration time.
//!
//! So a record here is a [`serde_json::Value`] object keyed by **field name**,
//! exactly as the workflow engine passes payloads, and a caller with a typed
//! struct converts at the boundary with `serde_json::to_value` /
//! `from_value`. That keeps both traits object-safe, so a
//! `Box<dyn VectorCollection>` can be swapped between providers — which a
//! generic-over-record-type trait could not be.
//!
//! The one piece of upstream's model layer that is *not* redundant is the
//! split between a field's logical name and its storage name, since that is a
//! property of the store rather than of the Rust type.
//! [`VectorStoreCollectionDefinition::to_storage`] and
//! [`VectorStoreCollectionDefinition::from_storage`] perform that renaming.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::error::{Error, Result};

/// How a vector index is built. Open value wrapper — the constants cover
/// upstream's list, and any other string a provider understands is accepted.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct IndexKind(pub String);

impl IndexKind {
    pub const HNSW: &'static str = "hnsw";
    pub const FLAT: &'static str = "flat";
    pub const IVF_FLAT: &'static str = "ivf_flat";
    pub const DISK_ANN: &'static str = "disk_ann";
    pub const QUANTIZED_FLAT: &'static str = "quantized_flat";
    pub const DYNAMIC: &'static str = "dynamic";
    pub const DEFAULT: &'static str = "default";

    pub fn new(value: impl Into<String>) -> Self {
        IndexKind(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// How similarity between two vectors is measured. Open value wrapper, as
/// [`IndexKind`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DistanceFunction(pub String);

impl DistanceFunction {
    pub const COSINE_SIMILARITY: &'static str = "cosine_similarity";
    pub const COSINE_DISTANCE: &'static str = "cosine_distance";
    pub const DOT_PROD: &'static str = "dot_prod";
    pub const EUCLIDEAN_DISTANCE: &'static str = "euclidean_distance";
    pub const EUCLIDEAN_SQUARED_DISTANCE: &'static str = "euclidean_squared_distance";
    pub const MANHATTAN: &'static str = "manhattan";
    pub const HAMMING: &'static str = "hamming";

    pub fn new(value: impl Into<String>) -> Self {
        DistanceFunction(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Whether a **higher** score means a closer match under this function.
    ///
    /// Similarity functions rank descending; distance functions rank
    /// ascending. Getting this backwards silently returns the *worst* matches
    /// first, which is why it belongs here rather than in each provider.
    /// Mirrors upstream's `DISTANCE_FUNCTION_DIRECTION_HELPER`. `None` for a
    /// function this port does not know the direction of — a caller must not
    /// guess.
    pub fn higher_is_closer(&self) -> Option<bool> {
        match self.0.as_str() {
            Self::COSINE_SIMILARITY | Self::DOT_PROD => Some(true),
            Self::COSINE_DISTANCE
            | Self::EUCLIDEAN_DISTANCE
            | Self::EUCLIDEAN_SQUARED_DISTANCE
            | Self::MANHATTAN
            | Self::HAMMING => Some(false),
            _ => None,
        }
    }
}

/// What role a field plays in a collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldType {
    /// The record's primary key. Exactly one per collection.
    Key,
    /// An ordinary payload field.
    Data,
    /// An embedding vector.
    Vector,
}

/// One field in a vector-store collection.
///
/// Build with [`VectorStoreField::key`], [`VectorStoreField::data`] or
/// [`VectorStoreField::vector`], then refine with the builder methods —
/// mirroring upstream's three `__init__` overloads, but with the
/// field-type-specific options only reachable on the variant that accepts
/// them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorStoreField {
    pub field_type: FieldType,
    /// The field's name on the record.
    pub name: String,
    /// The field's name in the store, when it differs from [`Self::name`].
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub storage_name: Option<String>,
    /// Provider-specific type hint (e.g. `"str"`, `"int"`), passed through to
    /// collection creation.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub type_: Option<String>,
    /// Whether the store should index this field for filtering.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub is_indexed: Option<bool>,
    /// Whether the store should index this field for full-text search.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub is_full_text_indexed: Option<bool>,
    /// Vector fields only: the embedding's dimensionality.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub dimensions: Option<usize>,
    /// Vector fields only.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub index_kind: Option<IndexKind>,
    /// Vector fields only.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub distance_function: Option<DistanceFunction>,
}

impl VectorStoreField {
    fn bare(field_type: FieldType, name: impl Into<String>) -> Self {
        Self {
            field_type,
            name: name.into(),
            storage_name: None,
            type_: None,
            is_indexed: None,
            is_full_text_indexed: None,
            dimensions: None,
            index_kind: None,
            distance_function: None,
        }
    }

    /// The collection's primary key field.
    pub fn key(name: impl Into<String>) -> Self {
        Self::bare(FieldType::Key, name)
    }

    /// An ordinary payload field.
    pub fn data(name: impl Into<String>) -> Self {
        Self::bare(FieldType::Data, name)
    }

    /// An embedding field of `dimensions` dimensions.
    pub fn vector(name: impl Into<String>, dimensions: usize) -> Self {
        let mut field = Self::bare(FieldType::Vector, name);
        field.dimensions = Some(dimensions);
        field
    }

    /// Store this field under a different name than the record uses.
    pub fn with_storage_name(mut self, storage_name: impl Into<String>) -> Self {
        self.storage_name = Some(storage_name.into());
        self
    }

    /// Provider-specific type hint.
    pub fn with_type(mut self, type_: impl Into<String>) -> Self {
        self.type_ = Some(type_.into());
        self
    }

    /// Index this field for filtering.
    pub fn indexed(mut self) -> Self {
        self.is_indexed = Some(true);
        self
    }

    /// Index this field for full-text search.
    pub fn full_text_indexed(mut self) -> Self {
        self.is_full_text_indexed = Some(true);
        self
    }

    /// Vector fields only; ignored by providers on other field types.
    pub fn with_index_kind(mut self, kind: IndexKind) -> Self {
        self.index_kind = Some(kind);
        self
    }

    /// Vector fields only; ignored by providers on other field types.
    pub fn with_distance_function(mut self, function: DistanceFunction) -> Self {
        self.distance_function = Some(function);
        self
    }

    /// The name this field is stored under: [`Self::storage_name`] when set,
    /// otherwise [`Self::name`].
    pub fn effective_storage_name(&self) -> &str {
        self.storage_name.as_deref().unwrap_or(&self.name)
    }
}

/// The shape of one vector-store collection: its fields, and the accessors a
/// provider needs to build requests from them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorStoreCollectionDefinition {
    pub fields: Vec<VectorStoreField>,
}

impl VectorStoreCollectionDefinition {
    /// Build and validate a definition.
    ///
    /// Validation is at construction rather than at first use because every
    /// failure it catches — no key, two keys, duplicate names, a vector field
    /// with no dimensions — produces a collection that cannot work at all,
    /// and a provider would otherwise surface it as an opaque error from the
    /// service on the first upsert. Mirrors upstream's `_validate`.
    pub fn new(fields: Vec<VectorStoreField>) -> Result<Self> {
        let definition = Self { fields };
        definition.validate()?;
        Ok(definition)
    }

    fn validate(&self) -> Result<()> {
        if self.fields.is_empty() {
            return Err(Error::Configuration(
                "a vector store collection definition needs at least one field".into(),
            ));
        }

        let keys: Vec<&VectorStoreField> = self
            .fields
            .iter()
            .filter(|f| f.field_type == FieldType::Key)
            .collect();
        match keys.len() {
            1 => {}
            0 => {
                return Err(Error::Configuration(
                    "a vector store collection definition needs exactly one key field, found none"
                        .into(),
                ))
            }
            n => {
                return Err(Error::Configuration(format!(
                    "a vector store collection definition needs exactly one key field, found {n}"
                )))
            }
        }

        let mut seen_names: Vec<&str> = Vec::with_capacity(self.fields.len());
        let mut seen_storage: Vec<&str> = Vec::with_capacity(self.fields.len());
        for field in &self.fields {
            if field.name.is_empty() {
                return Err(Error::Configuration(
                    "a vector store field name must not be empty".into(),
                ));
            }
            if seen_names.contains(&field.name.as_str()) {
                return Err(Error::Configuration(format!(
                    "duplicate vector store field name '{}'",
                    field.name
                )));
            }
            seen_names.push(&field.name);

            // Two fields renamed onto one storage name would silently
            // overwrite each other on every write.
            let storage = field.effective_storage_name();
            if seen_storage.contains(&storage) {
                return Err(Error::Configuration(format!(
                    "two vector store fields map to the same storage name '{storage}'"
                )));
            }
            seen_storage.push(storage);

            if field.field_type == FieldType::Vector {
                match field.dimensions {
                    Some(d) if d > 0 => {}
                    _ => {
                        return Err(Error::Configuration(format!(
                            "vector field '{}' must declare a non-zero number of dimensions",
                            field.name
                        )))
                    }
                }
            }
        }
        Ok(())
    }

    /// Every field name, in declaration order.
    pub fn names(&self) -> Vec<&str> {
        self.fields.iter().map(|f| f.name.as_str()).collect()
    }

    /// Every storage name, in declaration order.
    pub fn storage_names(&self) -> Vec<&str> {
        self.fields
            .iter()
            .map(VectorStoreField::effective_storage_name)
            .collect()
    }

    /// The key field. Infallible: [`Self::new`] rejects a definition without
    /// exactly one.
    pub fn key_field(&self) -> &VectorStoreField {
        self.fields
            .iter()
            .find(|f| f.field_type == FieldType::Key)
            .expect("validated at construction: exactly one key field")
    }

    /// The key field's storage name.
    pub fn key_field_storage_name(&self) -> &str {
        self.key_field().effective_storage_name()
    }

    /// Every vector field.
    pub fn vector_fields(&self) -> Vec<&VectorStoreField> {
        self.fields
            .iter()
            .filter(|f| f.field_type == FieldType::Vector)
            .collect()
    }

    /// Every data field.
    pub fn data_fields(&self) -> Vec<&VectorStoreField> {
        self.fields
            .iter()
            .filter(|f| f.field_type == FieldType::Data)
            .collect()
    }

    /// Look up a vector field by name, or the only one when `name` is `None`.
    ///
    /// Returns `None` when the name does not match a vector field, and also
    /// when `name` is `None` and the collection has several — there is no
    /// safe default to pick, and searching the wrong embedding returns
    /// confident nonsense rather than an error.
    pub fn try_get_vector_field(&self, name: Option<&str>) -> Option<&VectorStoreField> {
        let vectors = self.vector_fields();
        match name {
            Some(name) => vectors.into_iter().find(|f| f.name == name),
            None => match vectors.as_slice() {
                [only] => Some(only),
                _ => None,
            },
        }
    }

    /// Field names, filtered. Mirrors upstream's `get_names`.
    pub fn get_names(&self, include_vector_fields: bool, include_key_field: bool) -> Vec<&str> {
        self.selected(include_vector_fields, include_key_field)
            .map(|f| f.name.as_str())
            .collect()
    }

    /// Storage names, filtered. Mirrors upstream's `get_storage_names`.
    pub fn get_storage_names(
        &self,
        include_vector_fields: bool,
        include_key_field: bool,
    ) -> Vec<&str> {
        self.selected(include_vector_fields, include_key_field)
            .map(VectorStoreField::effective_storage_name)
            .collect()
    }

    fn selected(
        &self,
        include_vector_fields: bool,
        include_key_field: bool,
    ) -> impl Iterator<Item = &VectorStoreField> {
        self.fields.iter().filter(move |f| match f.field_type {
            FieldType::Vector => include_vector_fields,
            FieldType::Key => include_key_field,
            FieldType::Data => true,
        })
    }

    /// Rename a record's fields from logical names to storage names.
    ///
    /// Fields the definition does not declare are dropped rather than passed
    /// through, so a stray key cannot reach the store and be rejected (or,
    /// worse, silently persisted outside the schema).
    pub fn to_storage(&self, record: &Value) -> Result<Value> {
        let source = record.as_object().ok_or_else(|| {
            Error::Configuration("a vector store record must be a JSON object".into())
        })?;
        let mut out = Map::new();
        for field in &self.fields {
            if let Some(value) = source.get(&field.name) {
                out.insert(field.effective_storage_name().to_string(), value.clone());
            }
        }
        Ok(Value::Object(out))
    }

    /// Rename a stored record's fields back to logical names, optionally
    /// dropping vector fields (which are large and rarely wanted on read).
    pub fn from_storage(&self, stored: &Value, include_vectors: bool) -> Result<Value> {
        let source = stored.as_object().ok_or_else(|| {
            Error::Configuration("a stored vector record must be a JSON object".into())
        })?;
        let mut out = Map::new();
        for field in &self.fields {
            if !include_vectors && field.field_type == FieldType::Vector {
                continue;
            }
            if let Some(value) = source.get(field.effective_storage_name()) {
                out.insert(field.name.clone(), value.clone());
            }
        }
        Ok(Value::Object(out))
    }
}

/// Paging and shaping options for a search.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorSearchOptions {
    /// Maximum number of records to return. Must be greater than zero.
    pub top: usize,
    /// Number of leading records to skip.
    pub skip: usize,
    /// Whether returned records carry their vector fields.
    pub include_vectors: bool,
    /// Which vector field to search, when the collection has several.
    pub vector_field_name: Option<String>,
    /// Provider-specific filter expression, passed through verbatim.
    ///
    /// Upstream accepts a Python lambda and parses its AST into each
    /// provider's filter dialect. That has no Rust counterpart — there is no
    /// runtime AST to walk — so a filter here is the provider's own
    /// expression, and a provider documents its dialect.
    pub filter: Option<String>,
}

impl Default for VectorSearchOptions {
    fn default() -> Self {
        Self {
            // Upstream's default; an unbounded search against a large
            // collection is never what a caller meant.
            top: 10,
            skip: 0,
            include_vectors: false,
            vector_field_name: None,
            filter: None,
        }
    }
}

impl VectorSearchOptions {
    pub fn new(top: usize) -> Self {
        Self {
            top,
            ..Default::default()
        }
    }

    pub fn with_skip(mut self, skip: usize) -> Self {
        self.skip = skip;
        self
    }

    pub fn with_include_vectors(mut self, include: bool) -> Self {
        self.include_vectors = include;
        self
    }

    pub fn with_vector_field_name(mut self, name: impl Into<String>) -> Self {
        self.vector_field_name = Some(name.into());
        self
    }

    pub fn with_filter(mut self, filter: impl Into<String>) -> Self {
        self.filter = Some(filter.into());
        self
    }

    /// Reject options a provider cannot act on. Mirrors upstream's
    /// `_validate_paging`; `skip` needs no check because `usize` cannot be
    /// negative.
    pub fn validate(&self) -> Result<()> {
        if self.top == 0 {
            return Err(Error::Configuration(
                "vector search `top` must be greater than zero".into(),
            ));
        }
        Ok(())
    }
}

/// One search hit: the record, and the provider's score for it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorSearchResult {
    /// The record, keyed by logical field name.
    pub record: Value,
    /// The provider's similarity or distance score. Whether a higher value is
    /// a closer match depends on the collection's
    /// [`DistanceFunction::higher_is_closer`]; `None` when the provider
    /// reports no score.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub score: Option<f64>,
}

/// One collection's data plane: create/drop, read/write, and search.
///
/// Object-safe, so a caller can hold `Box<dyn VectorCollection>` and swap
/// providers. Records are `serde_json::Value` objects keyed by logical field
/// name — see the module docs.
#[async_trait::async_trait]
pub trait VectorCollection: Send + Sync {
    /// The collection's name in the store.
    fn name(&self) -> &str;

    /// The collection's field definition.
    fn definition(&self) -> &VectorStoreCollectionDefinition;

    /// Create the collection if it does not exist. Succeeds when it already
    /// does.
    async fn ensure_collection_exists(&self) -> Result<()>;

    /// Whether the collection exists.
    async fn collection_exists(&self) -> Result<bool>;

    /// Delete the collection if it exists. Succeeds when it does not.
    async fn ensure_collection_deleted(&self) -> Result<()>;

    /// Insert or replace `records`, returning their keys in input order.
    async fn upsert(&self, records: Vec<Value>) -> Result<Vec<Value>>;

    /// Fetch records by key, in the order the keys were given. A key with no
    /// record yields `None` in its slot rather than shortening the result, so
    /// a caller can align the two lists.
    async fn get(&self, keys: Vec<Value>, include_vectors: bool) -> Result<Vec<Option<Value>>>;

    /// Delete records by key. Deleting an absent key is not an error.
    async fn delete(&self, keys: Vec<Value>) -> Result<()>;

    /// Search by embedding.
    async fn search(
        &self,
        vector: Vec<f32>,
        options: &VectorSearchOptions,
    ) -> Result<Vec<VectorSearchResult>>;
}

/// A connection to a vector store: lists collections and hands them out.
#[async_trait::async_trait]
pub trait VectorStore: Send + Sync {
    /// Open a collection by name against `definition`.
    ///
    /// No I/O: this describes the collection rather than creating it. Call
    /// [`VectorCollection::ensure_collection_exists`] for that.
    fn get_collection(
        &self,
        name: &str,
        definition: VectorStoreCollectionDefinition,
    ) -> Result<Box<dyn VectorCollection>>;

    /// Every collection name in the store.
    async fn list_collection_names(&self) -> Result<Vec<String>>;

    /// Whether a collection exists. Defaults to a
    /// [`Self::list_collection_names`] scan, which every store supports;
    /// override where the provider has a cheaper direct check.
    async fn collection_exists(&self, name: &str) -> Result<bool> {
        Ok(self
            .list_collection_names()
            .await?
            .iter()
            .any(|n| n == name))
    }
}

/// An in-memory [`VectorStore`], for tests and for running a pipeline before
/// a real store is provisioned.
///
/// Brute-force search: every record's vector is scored against the query, so
/// cost is linear in collection size. That is the right trade for a few
/// thousand records and the wrong one for a few million — this is not a
/// substitute for a real index.
#[derive(Default)]
pub struct InMemoryVectorStore {
    /// Shared with every collection handle this store hands out, so two
    /// handles to one name address the same records rather than diverging
    /// copies.
    collections: std::sync::Arc<std::sync::Mutex<HashMap<String, InMemoryData>>>,
}

#[derive(Default)]
struct InMemoryData {
    exists: bool,
    /// Keyed by the record's key rendered as a string, so any JSON scalar key
    /// works without requiring `Value: Hash`.
    records: HashMap<String, Value>,
}

impl InMemoryVectorStore {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Render a key `Value` as the map key used by [`InMemoryVectorStore`].
/// A JSON string key uses its contents, so `"a"` and `a` address one record.
fn key_string(key: &Value) -> String {
    match key {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Cosine similarity, or `None` when either vector is zero-length or the
/// lengths differ (a dimension mismatch is a schema error, not a distant
/// match).
fn cosine_similarity(a: &[f32], b: &[f32]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        dot += f64::from(*x) * f64::from(*y);
        na += f64::from(*x) * f64::from(*x);
        nb += f64::from(*y) * f64::from(*y);
    }
    if na == 0.0 || nb == 0.0 {
        return None;
    }
    Some(dot / (na.sqrt() * nb.sqrt()))
}

struct InMemoryCollection {
    name: String,
    definition: VectorStoreCollectionDefinition,
    store: std::sync::Arc<std::sync::Mutex<HashMap<String, InMemoryData>>>,
}

impl InMemoryCollection {
    fn with_data<R>(&self, f: impl FnOnce(&mut InMemoryData) -> R) -> R {
        let mut guard = self.store.lock().expect("in-memory vector store poisoned");
        f(guard.entry(self.name.clone()).or_default())
    }
}

#[async_trait::async_trait]
impl VectorCollection for InMemoryCollection {
    fn name(&self) -> &str {
        &self.name
    }

    fn definition(&self) -> &VectorStoreCollectionDefinition {
        &self.definition
    }

    async fn ensure_collection_exists(&self) -> Result<()> {
        self.with_data(|d| d.exists = true);
        Ok(())
    }

    async fn collection_exists(&self) -> Result<bool> {
        Ok(self.with_data(|d| d.exists))
    }

    async fn ensure_collection_deleted(&self) -> Result<()> {
        self.with_data(|d| {
            d.exists = false;
            d.records.clear();
        });
        Ok(())
    }

    async fn upsert(&self, records: Vec<Value>) -> Result<Vec<Value>> {
        let key_name = self.definition.key_field().name.clone();
        let mut keys = Vec::with_capacity(records.len());
        for record in &records {
            let key = record.get(&key_name).cloned().ok_or_else(|| {
                Error::Configuration(format!("record is missing its key field '{key_name}'"))
            })?;
            keys.push(key);
        }
        self.with_data(|d| {
            for (key, record) in keys.iter().zip(records.iter()) {
                d.records.insert(key_string(key), record.clone());
            }
        });
        Ok(keys)
    }

    async fn get(&self, keys: Vec<Value>, include_vectors: bool) -> Result<Vec<Option<Value>>> {
        let stored: Vec<Option<Value>> = self.with_data(|d| {
            keys.iter()
                .map(|k| d.records.get(&key_string(k)).cloned())
                .collect()
        });
        stored
            .into_iter()
            .map(|record| match record {
                Some(r) if !include_vectors => {
                    // Records are already keyed logically here, so the
                    // round trip exists only to drop the vector fields.
                    self.definition.from_storage(&r, false).map(Some)
                }
                other => Ok(other),
            })
            .collect()
    }

    async fn delete(&self, keys: Vec<Value>) -> Result<()> {
        self.with_data(|d| {
            for key in &keys {
                d.records.remove(&key_string(key));
            }
        });
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
                Error::Configuration(
                    "no vector field to search: name it with `vector_field_name` when the \
                     collection declares more than one"
                        .into(),
                )
            })?;
        let field_name = field.name.clone();

        let records: Vec<Value> = self.with_data(|d| d.records.values().cloned().collect());
        let mut scored: Vec<(f64, Value)> = Vec::new();
        for record in records {
            let Some(stored_vector) = record.get(&field_name).and_then(Value::as_array) else {
                continue;
            };
            let stored: Vec<f32> = stored_vector
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect();
            if let Some(score) = cosine_similarity(&vector, &stored) {
                scored.push((score, record));
            }
        }
        // Cosine similarity: higher is closer.
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));

        scored
            .into_iter()
            .skip(options.skip)
            .take(options.top)
            .map(|(score, record)| {
                Ok(VectorSearchResult {
                    record: self
                        .definition
                        .from_storage(&record, options.include_vectors)?,
                    score: Some(score),
                })
            })
            .collect()
    }
}

#[async_trait::async_trait]
impl VectorStore for InMemoryVectorStore {
    fn get_collection(
        &self,
        name: &str,
        definition: VectorStoreCollectionDefinition,
    ) -> Result<Box<dyn VectorCollection>> {
        Ok(Box::new(InMemoryCollection {
            name: name.to_string(),
            definition,
            store: std::sync::Arc::clone(&self.collections),
        }))
    }

    async fn list_collection_names(&self) -> Result<Vec<String>> {
        Ok(self
            .collections
            .lock()
            .expect("in-memory vector store poisoned")
            .iter()
            .filter(|(_, d)| d.exists)
            .map(|(name, _)| name.clone())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn definition() -> VectorStoreCollectionDefinition {
        VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id"),
            VectorStoreField::data("text").full_text_indexed(),
            VectorStoreField::vector("embedding", 3)
                .with_distance_function(DistanceFunction::new(DistanceFunction::COSINE_SIMILARITY)),
        ])
        .unwrap()
    }

    // region: definition validation

    #[test]
    fn a_definition_needs_exactly_one_key() {
        let none = VectorStoreCollectionDefinition::new(vec![VectorStoreField::data("text")]);
        assert!(none.unwrap_err().to_string().contains("found none"));

        let two = VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("a"),
            VectorStoreField::key("b"),
        ]);
        assert!(two.unwrap_err().to_string().contains("found 2"));
    }

    #[test]
    fn duplicate_field_names_are_rejected() {
        let err = VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id"),
            VectorStoreField::data("text"),
            VectorStoreField::data("text"),
        ])
        .unwrap_err();
        assert!(err.to_string().contains("duplicate"), "{err}");
    }

    #[test]
    fn two_fields_sharing_a_storage_name_are_rejected() {
        // They would overwrite each other on every write, which the store
        // would report as nothing at all.
        let err = VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id"),
            VectorStoreField::data("title").with_storage_name("t"),
            VectorStoreField::data("text").with_storage_name("t"),
        ])
        .unwrap_err();
        assert!(err.to_string().contains("same storage name"), "{err}");
    }

    #[test]
    fn a_vector_field_needs_dimensions() {
        let mut field = VectorStoreField::vector("embedding", 3);
        field.dimensions = None;
        let err =
            VectorStoreCollectionDefinition::new(vec![VectorStoreField::key("id"), field.clone()])
                .unwrap_err();
        assert!(err.to_string().contains("dimensions"), "{err}");

        field.dimensions = Some(0);
        let err = VectorStoreCollectionDefinition::new(vec![VectorStoreField::key("id"), field])
            .unwrap_err();
        assert!(err.to_string().contains("dimensions"), "{err}");
    }

    #[test]
    fn an_empty_definition_is_rejected() {
        assert!(VectorStoreCollectionDefinition::new(vec![]).is_err());
    }

    // endregion

    // region: accessors

    #[test]
    fn accessors_report_the_declared_shape() {
        let d = definition();
        assert_eq!(d.names(), vec!["id", "text", "embedding"]);
        assert_eq!(d.key_field().name, "id");
        assert_eq!(d.key_field_storage_name(), "id");
        assert_eq!(d.vector_fields().len(), 1);
        assert_eq!(d.data_fields().len(), 1);
        assert_eq!(d.get_names(false, false), vec!["text"]);
        assert_eq!(d.get_names(true, false), vec!["text", "embedding"]);
        assert_eq!(d.get_names(false, true), vec!["id", "text"]);
    }

    #[test]
    fn storage_names_follow_the_override() {
        let d = VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id").with_storage_name("_id"),
            VectorStoreField::data("text"),
        ])
        .unwrap();
        assert_eq!(d.storage_names(), vec!["_id", "text"]);
        assert_eq!(d.key_field_storage_name(), "_id");
        assert_eq!(d.get_storage_names(true, true), vec!["_id", "text"]);
    }

    #[test]
    fn a_single_vector_field_is_found_without_naming_it() {
        assert_eq!(
            definition().try_get_vector_field(None).map(|f| &f.name),
            Some(&"embedding".to_string())
        );
    }

    #[test]
    fn several_vector_fields_must_be_named_explicitly() {
        // Picking one arbitrarily would search the wrong embedding and return
        // confident nonsense, so there is deliberately no default.
        let d = VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id"),
            VectorStoreField::vector("title_vec", 3),
            VectorStoreField::vector("body_vec", 3),
        ])
        .unwrap();
        assert!(d.try_get_vector_field(None).is_none());
        assert_eq!(
            d.try_get_vector_field(Some("body_vec")).map(|f| &f.name),
            Some(&"body_vec".to_string())
        );
        assert!(d.try_get_vector_field(Some("nope")).is_none());
    }

    // endregion

    // region: storage-name mapping

    #[test]
    fn to_storage_renames_and_drops_undeclared_fields() {
        let d = VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id").with_storage_name("_id"),
            VectorStoreField::data("text"),
        ])
        .unwrap();
        let out = d
            .to_storage(&json!({"id": "1", "text": "hi", "stray": true}))
            .unwrap();
        assert_eq!(out, json!({"_id": "1", "text": "hi"}));
    }

    #[test]
    fn from_storage_renames_back_and_can_drop_vectors() {
        let d = definition();
        let stored = json!({"id": "1", "text": "hi", "embedding": [1.0, 0.0, 0.0]});
        assert_eq!(
            d.from_storage(&stored, false).unwrap(),
            json!({"id": "1", "text": "hi"})
        );
        assert_eq!(d.from_storage(&stored, true).unwrap(), stored);
    }

    #[test]
    fn mapping_rejects_a_non_object_record() {
        assert!(definition().to_storage(&json!([1, 2])).is_err());
        assert!(definition().from_storage(&json!("x"), true).is_err());
    }

    // endregion

    // region: distance direction

    #[test]
    fn distance_direction_is_known_for_the_documented_functions() {
        assert_eq!(
            DistanceFunction::new(DistanceFunction::COSINE_SIMILARITY).higher_is_closer(),
            Some(true)
        );
        assert_eq!(
            DistanceFunction::new(DistanceFunction::DOT_PROD).higher_is_closer(),
            Some(true)
        );
        for f in [
            DistanceFunction::COSINE_DISTANCE,
            DistanceFunction::EUCLIDEAN_DISTANCE,
            DistanceFunction::EUCLIDEAN_SQUARED_DISTANCE,
            DistanceFunction::MANHATTAN,
            DistanceFunction::HAMMING,
        ] {
            assert_eq!(
                DistanceFunction::new(f).higher_is_closer(),
                Some(false),
                "{f}"
            );
        }
        // An unknown function must not be guessed at.
        assert_eq!(DistanceFunction::new("bespoke").higher_is_closer(), None);
    }

    // endregion

    // region: search options

    #[test]
    fn search_options_reject_a_zero_top() {
        assert!(VectorSearchOptions::new(0).validate().is_err());
        assert!(VectorSearchOptions::new(1).validate().is_ok());
        assert_eq!(VectorSearchOptions::default().top, 10);
        assert!(!VectorSearchOptions::default().include_vectors);
    }

    // endregion

    // region: in-memory store

    fn record(id: &str, text: &str, embedding: [f32; 3]) -> Value {
        json!({"id": id, "text": text, "embedding": embedding})
    }

    #[tokio::test]
    async fn in_memory_round_trips_records() {
        let store = InMemoryVectorStore::new();
        let c = store.get_collection("docs", definition()).unwrap();
        c.ensure_collection_exists().await.unwrap();
        assert!(c.collection_exists().await.unwrap());

        let keys = c
            .upsert(vec![
                record("a", "alpha", [1.0, 0.0, 0.0]),
                record("b", "beta", [0.0, 1.0, 0.0]),
            ])
            .await
            .unwrap();
        assert_eq!(keys, vec![json!("a"), json!("b")]);

        let got = c.get(vec![json!("a")], true).await.unwrap();
        assert_eq!(got[0].as_ref().unwrap()["text"], json!("alpha"));

        // A missing key holds its slot so the two lists stay aligned.
        let got = c.get(vec![json!("a"), json!("zz")], false).await.unwrap();
        assert_eq!(got.len(), 2);
        assert!(got[1].is_none());
        // include_vectors: false drops the embedding.
        assert!(got[0].as_ref().unwrap().get("embedding").is_none());

        c.delete(vec![json!("a")]).await.unwrap();
        assert!(c.get(vec![json!("a")], true).await.unwrap()[0].is_none());
    }

    #[tokio::test]
    async fn in_memory_search_ranks_by_cosine_similarity() {
        let store = InMemoryVectorStore::new();
        let c = store.get_collection("docs", definition()).unwrap();
        c.ensure_collection_exists().await.unwrap();
        c.upsert(vec![
            record("a", "alpha", [1.0, 0.0, 0.0]),
            record("b", "beta", [0.0, 1.0, 0.0]),
            record("c", "gamma", [0.9, 0.1, 0.0]),
        ])
        .await
        .unwrap();

        let hits = c
            .search(vec![1.0, 0.0, 0.0], &VectorSearchOptions::new(2))
            .await
            .unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].record["id"], json!("a"));
        assert_eq!(hits[1].record["id"], json!("c"));
        assert!(hits[0].score.unwrap() >= hits[1].score.unwrap());
        // Vectors withheld by default.
        assert!(hits[0].record.get("embedding").is_none());
    }

    #[tokio::test]
    async fn in_memory_search_honors_top_skip_and_include_vectors() {
        let store = InMemoryVectorStore::new();
        let c = store.get_collection("docs", definition()).unwrap();
        c.upsert(vec![
            record("a", "alpha", [1.0, 0.0, 0.0]),
            record("c", "gamma", [0.9, 0.1, 0.0]),
        ])
        .await
        .unwrap();

        let hits = c
            .search(
                vec![1.0, 0.0, 0.0],
                &VectorSearchOptions::new(5)
                    .with_skip(1)
                    .with_include_vectors(true),
            )
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record["id"], json!("c"));
        assert!(hits[0].record.get("embedding").is_some());
    }

    #[tokio::test]
    async fn upsert_without_the_key_field_is_an_error() {
        let store = InMemoryVectorStore::new();
        let c = store.get_collection("docs", definition()).unwrap();
        let err = c
            .upsert(vec![json!({"text": "no key here"})])
            .await
            .unwrap_err();
        assert!(err.to_string().contains("key field"), "{err}");
    }

    #[tokio::test]
    async fn a_dimension_mismatch_is_skipped_rather_than_scored() {
        // A wrong-length vector is a schema error, not a distant match, so it
        // must not be ranked at all.
        let store = InMemoryVectorStore::new();
        let c = store.get_collection("docs", definition()).unwrap();
        c.upsert(vec![
            json!({"id": "x", "text": "t", "embedding": [1.0, 0.0]}),
        ])
        .await
        .unwrap();
        let hits = c
            .search(vec![1.0, 0.0, 0.0], &VectorSearchOptions::new(5))
            .await
            .unwrap();
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn search_without_naming_a_field_fails_when_several_exist() {
        let d = VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id"),
            VectorStoreField::vector("title_vec", 2),
            VectorStoreField::vector("body_vec", 2),
        ])
        .unwrap();
        let store = InMemoryVectorStore::new();
        let c = store.get_collection("docs", d).unwrap();
        let err = c
            .search(vec![1.0, 0.0], &VectorSearchOptions::new(1))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("vector_field_name"), "{err}");
    }

    #[tokio::test]
    async fn deleting_a_collection_clears_it() {
        let store = InMemoryVectorStore::new();
        let c = store.get_collection("docs", definition()).unwrap();
        c.ensure_collection_exists().await.unwrap();
        c.upsert(vec![record("a", "alpha", [1.0, 0.0, 0.0])])
            .await
            .unwrap();
        c.ensure_collection_deleted().await.unwrap();
        assert!(!c.collection_exists().await.unwrap());
        assert!(c.get(vec![json!("a")], true).await.unwrap()[0].is_none());
    }

    #[tokio::test]
    async fn the_trait_pair_is_object_safe() {
        // The point of `Value` records: a caller can hold provider-agnostic
        // handles.
        let store: Box<dyn VectorStore> = Box::new(InMemoryVectorStore::new());
        let c: Box<dyn VectorCollection> = store.get_collection("docs", definition()).unwrap();
        c.ensure_collection_exists().await.unwrap();
        assert_eq!(c.name(), "docs");
        assert_eq!(c.definition().key_field().name, "id");
        assert!(store.collection_exists("docs").await.unwrap());
        assert!(!store.collection_exists("other").await.unwrap());
    }

    // endregion
}
