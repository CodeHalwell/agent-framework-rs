//! Azure Cosmos DB for NoSQL as a [`VectorStore`].
//!
//! A container becomes a vector collection: its documents are the records,
//! its `vectorEmbeddingPolicy` declares the embeddings, and `VectorDistance`
//! in a SQL query is the search. Speaks the Cosmos DB REST API directly
//! through the same signed [`CosmosRestClient`] the history and checkpoint
//! stores use, so both authentication modes (master key and Microsoft Entra
//! ID) are available here unchanged.
//!
//! ```no_run
//! use agent_framework_core::vectors::{
//!     DistanceFunction, VectorCollection, VectorSearchOptions,
//!     VectorStoreCollectionDefinition, VectorStoreField,
//! };
//! use agent_framework_cosmos::CosmosVectorStore;
//! use serde_json::json;
//!
//! # async fn demo() -> agent_framework_core::error::Result<()> {
//! let definition = VectorStoreCollectionDefinition::new(vec![
//!     VectorStoreField::key("id").with_type("str"),
//!     VectorStoreField::data("text").with_type("str"),
//!     VectorStoreField::vector("embedding", 1536)
//!         .with_distance_function(DistanceFunction::new(DistanceFunction::COSINE_SIMILARITY)),
//! ])?;
//!
//! let store = CosmosVectorStore::new(
//!     "https://my-account.documents.azure.com:443/",
//!     "<base64 master key>",
//!     "agent-framework",
//! )?;
//! let collection = store.collection("memories", definition)?;
//! collection.ensure_collection_exists().await?;
//! collection
//!     .upsert(vec![json!({"id": "m1", "text": "hello", "embedding": vec![0.0f32; 1536]})])
//!     .await?;
//! let hits = collection
//!     .search(vec![0.0; 1536], &VectorSearchOptions::new(5))
//!     .await?;
//! # let _ = hits;
//! # Ok(())
//! # }
//! ```
//!
//! # What this connector does not carry
//!
//! Upstream reaches three groups of Cosmos-specific knobs through a
//! `provider_annotations` bag on a field and an `operation_options` mapping
//! on a call: the vector index's quantizer (`quantizer_type`,
//! `quantization_byte_size`, `indexing_search_list_size`) and the search's
//! `searchListSizeMultiplier` / `quantizedVectorListMultiplier` /
//! `filterPriority` / brute-force override. Neither bag exists on this
//! port's [`VectorStoreField`] or [`VectorSearchOptions`], and adding one for
//! a single connector would put an untyped escape hatch on a shared type. The
//! policies this builds are the service defaults for the chosen index kind;
//! a caller who needs the tuning knobs can create the container out of band
//! with them set and point [`CosmosVectorCollection`] at it — the policy
//! validation below accepts any container whose vector policy matches the
//! definition, and does not compare tuning options it did not ask for.

use std::collections::HashMap;
use std::sync::Arc;

use agent_framework_azure::TokenCredential;
use agent_framework_core::error::{Error, Result};
use agent_framework_core::vectors::{
    DistanceFunction, FieldType, Filter, FilterExpression, FilterGroupOperator, FilterOperator,
    IndexKind, VectorCollection, VectorSearchOptions, VectorSearchResult, VectorStore,
    VectorStoreCollectionDefinition, VectorStoreField,
};
use serde_json::{json, Map, Value};

use crate::client::{CosmosRestClient, DEFAULT_VECTOR_API_VERSION};

/// The partition key path every vector container this module creates uses.
///
/// `/id` makes each record its own logical partition, which is what makes a
/// point read by key a single-partition request and keeps writes spread
/// evenly. It also means a vector query is inherently cross-partition, which
/// is the normal shape for a similarity search: there is no partition to
/// narrow to when the query is "which vectors are nearest".
pub const VECTOR_PARTITION_KEY_PATH: &str = "/id";

/// Cosmos DB's per-item size limit. A record above it is rejected by the
/// service with a `413`, so it is caught here with the input index instead.
const ITEM_SIZE_LIMIT: usize = 2 * 1024 * 1024;

/// Cosmos DB's limit on an item's `id`, in UTF-8 bytes.
const ID_BYTE_LIMIT: usize = 1023;

/// The largest integer a JSON number round-trips exactly (2^53 − 1). Cosmos
/// stores numbers as IEEE 754 binary64, so a larger integer comes back
/// changed.
const MAX_SAFE_INTEGER: i64 = 9_007_199_254_740_991;

/// Cosmos DB's maximum JSON nesting depth for an item.
const MAX_JSON_DEPTH: usize = 128;

/// The dimension ceiling for a `flat` vector index; every other kind allows
/// 4,096.
const FLAT_MAX_DIMENSIONS: usize = 505;
const MAX_DIMENSIONS: usize = 4096;

// region: schema preparation

/// One vector field, resolved from the portable definition into the three
/// Cosmos-specific values the policies and the query need.
#[derive(Debug, Clone)]
struct VectorConfig {
    storage_name: String,
    path: String,
    data_type: &'static str,
    distance: &'static str,
    index_kind: &'static str,
}

/// A field's declared type, as far as this connector can tell from the
/// free-form [`VectorStoreField::type_`] hint.
///
/// Upstream reads a Python annotation, so a field always has a type; here the
/// hint is optional, and [`DeclaredType::Unspecified`] is the common case
/// rather than an error. Where upstream refuses an operation on an untyped
/// field, this connector emits the equivalent SQL type guard instead — see
/// [`CosmosVectorCollection::translate_condition`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeclaredType {
    Str,
    Int,
    Float,
    Bool,
    List,
    Object,
    Unspecified,
}

impl DeclaredType {
    fn of(field: &VectorStoreField) -> Self {
        match field.type_.as_deref() {
            None => Self::Unspecified,
            Some(hint) => match hint.trim().to_ascii_lowercase().as_str() {
                "str" | "string" => Self::Str,
                "int" | "integer" | "int32" | "int64" => Self::Int,
                "float" | "double" | "number" => Self::Float,
                "bool" | "boolean" => Self::Bool,
                "list" | "tuple" | "set" | "sequence" | "array" => Self::List,
                "dict" | "object" | "map" => Self::Object,
                _ => Self::Unspecified,
            },
        }
    }

    /// Whether `value` can be stored in, or compared against, a field of this
    /// type. `Unspecified` accepts anything — there is nothing to contradict.
    fn accepts(self, value: &Value) -> bool {
        match self {
            Self::Unspecified => true,
            Self::Str => value.is_string(),
            Self::Bool => value.is_boolean(),
            // A JSON integer is a valid float value; the reverse is not true.
            Self::Int => value.is_i64() || value.is_u64(),
            Self::Float => value.is_number(),
            Self::List => value.is_array(),
            Self::Object => value.is_object(),
        }
    }

    /// The SQL type-test function that holds for `value`'s JSON type, used as
    /// a guard when the field declares no type of its own.
    fn guard_for(value: &Value) -> Option<&'static str> {
        match value {
            Value::String(_) => Some("IS_STRING"),
            Value::Number(_) => Some("IS_NUMBER"),
            Value::Bool(_) => Some("IS_BOOL"),
            _ => None,
        }
    }
}

/// Map the portable distance function onto Cosmos DB's three.
///
/// A function Cosmos cannot compute is refused rather than approximated: a
/// silent substitution would rank by a different metric than the caller
/// declared, and a caller reading [`DistanceFunction::higher_is_closer`] off
/// their own definition would then have the direction wrong as well. The
/// three Cosmos does have each agree with the portable name's direction —
/// `cosine` and `dotproduct` are similarities (higher is closer), `euclidean`
/// is a distance (lower is closer) — which is why a search result here needs
/// no `score_kind` override.
fn cosmos_distance(field: &VectorStoreField) -> Result<&'static str> {
    let declared = field
        .distance_function
        .as_ref()
        .map(DistanceFunction::as_str)
        .unwrap_or(DistanceFunction::COSINE_SIMILARITY);
    match declared {
        DistanceFunction::COSINE_SIMILARITY | "cosine" => Ok("cosine"),
        DistanceFunction::DOT_PROD | "dotproduct" | "dot_product" => Ok("dotproduct"),
        DistanceFunction::EUCLIDEAN_DISTANCE | "euclidean" => Ok("euclidean"),
        DistanceFunction::COSINE_DISTANCE => Err(Error::Configuration(format!(
            "Azure Cosmos DB computes cosine as a similarity, not a distance, so vector field \
             '{}' cannot declare '{}'; declare '{}' instead — the ranking is identical and the \
             direction of the returned score is then correct",
            field.name,
            DistanceFunction::COSINE_DISTANCE,
            DistanceFunction::COSINE_SIMILARITY
        ))),
        other => Err(Error::Configuration(format!(
            "Azure Cosmos DB does not support the distance function '{other}' on vector field \
             '{}'; it computes cosine, dotproduct, or euclidean",
            field.name
        ))),
    }
}

/// Map the portable index kind onto Cosmos DB's three vector index types.
fn cosmos_index_kind(field: &VectorStoreField) -> Result<&'static str> {
    let declared = field
        .index_kind
        .as_ref()
        .map(IndexKind::as_str)
        .unwrap_or(IndexKind::DEFAULT);
    match declared {
        IndexKind::DEFAULT | IndexKind::QUANTIZED_FLAT | "quantizedFlat" => Ok("quantizedFlat"),
        IndexKind::FLAT => Ok("flat"),
        IndexKind::DISK_ANN | "diskANN" => Ok("diskANN"),
        other => Err(Error::Configuration(format!(
            "Azure Cosmos DB does not support the vector index kind '{other}' on field '{}'; it \
             offers flat, quantized_flat, and disk_ann",
            field.name
        ))),
    }
}

/// Cosmos vector paths address a top-level property, so a vector field's
/// storage name has to be one — the policy path is `/{name}`, and a name
/// carrying `/` or `"` would either address something else or not parse.
fn vector_storage_name(field: &VectorStoreField) -> Result<String> {
    let name = field.effective_storage_name();
    let ok = !name.is_empty()
        && name
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !ok {
        return Err(Error::Configuration(format!(
            "Azure Cosmos DB vector field '{}' must be stored under a top-level ASCII identifier \
             (letters, digits, and underscore, not starting with a digit), not '{name}'",
            field.name
        )));
    }
    Ok(name.to_string())
}

/// Resolve one vector field into its [`VectorConfig`].
fn prepare_vector_config(field: &VectorStoreField) -> Result<VectorConfig> {
    let storage_name = vector_storage_name(field)?;
    let data_type = match field.type_.as_deref() {
        None | Some("float") | Some("float32") => "float32",
        Some("int8") => "int8",
        Some("uint8") => "uint8",
        Some(other) => {
            return Err(Error::Configuration(format!(
                "Azure Cosmos DB vector field '{}' must hold float32, int8, or uint8 elements, \
                 not '{other}'",
                field.name
            )))
        }
    };
    let distance = cosmos_distance(field)?;
    let index_kind = cosmos_index_kind(field)?;
    let dimensions = field.dimensions.ok_or_else(|| {
        Error::Configuration(format!(
            "Azure Cosmos DB vector field '{}' must declare its dimensions",
            field.name
        ))
    })?;
    let maximum = if index_kind == "flat" {
        FLAT_MAX_DIMENSIONS
    } else {
        MAX_DIMENSIONS
    };
    if dimensions == 0 || dimensions > maximum {
        return Err(Error::Configuration(format!(
            "Azure Cosmos DB supports 1..={maximum} dimensions for a {index_kind} vector index, \
             and field '{}' declares {dimensions}",
            field.name
        )));
    }
    Ok(VectorConfig {
        path: format!("/{storage_name}"),
        storage_name,
        data_type,
        distance,
        index_kind,
    })
}

/// The indexing and vector-embedding policies for a definition, plus the
/// per-field configs the query builder needs.
#[derive(Debug)]
struct Schema {
    vector_policy: Value,
    indexing_policy: Value,
    configs: HashMap<String, VectorConfig>,
}

fn prepare_schema(definition: &VectorStoreCollectionDefinition) -> Result<Schema> {
    let key = definition.key_field();
    if key.effective_storage_name() != "id" {
        return Err(Error::Configuration(format!(
            "an Azure Cosmos DB container addresses items by their `id` property, so the key \
             field must be stored as 'id'; '{}' is stored as '{}' (use \
             `VectorStoreField::key(\"{}\").with_storage_name(\"id\")` to keep the record-side \
             name)",
            key.name,
            key.effective_storage_name(),
            key.name
        )));
    }
    if !DeclaredType::of(key).accepts(&Value::String(String::new())) {
        return Err(Error::Configuration(format!(
            "an Azure Cosmos DB item `id` is a string, so key field '{}' cannot declare type \
             '{}'",
            key.name,
            key.type_.as_deref().unwrap_or("")
        )));
    }
    if definition.vector_fields().is_empty() {
        return Err(Error::Configuration(
            "an Azure Cosmos DB vector collection needs at least one vector field".into(),
        ));
    }

    let mut configs = HashMap::new();
    let mut embeddings = Vec::new();
    let mut vector_indexes = Vec::new();
    // `_etag` is excluded by every Cosmos indexing policy: it changes on
    // every write and indexing it costs RUs for a value nothing filters on.
    let mut excluded_paths = vec![json!({ "path": "/_etag/?" })];

    for field in definition.fields() {
        if field.field_type != FieldType::Vector {
            if field.is_full_text_indexed == Some(true) {
                return Err(Error::Configuration(format!(
                    "this connector does not offer full-text or hybrid search, so field '{}' \
                     cannot be full-text indexed",
                    field.name
                )));
            }
            if field.field_type == FieldType::Data && field.is_indexed == Some(false) {
                excluded_paths
                    .push(json!({ "path": policy_path(field.effective_storage_name(), "/*") }));
            }
            continue;
        }
        let config = prepare_vector_config(field)?;
        embeddings.push(json!({
            "path": config.path,
            "dataType": config.data_type,
            "distanceFunction": config.distance,
            "dimensions": field.dimensions,
        }));
        vector_indexes.push(json!({ "path": config.path, "type": config.index_kind }));
        // A vector is excluded from the *ordinary* index: it is served by the
        // vector index beside it, and range-indexing 1,536 floats per
        // document is pure write amplification.
        excluded_paths.push(json!({ "path": format!("{}/*", config.path) }));
        configs.insert(field.name.clone(), config);
    }

    Ok(Schema {
        vector_policy: json!({ "vectorEmbeddings": embeddings }),
        indexing_policy: json!({
            "indexingMode": "consistent",
            "automatic": true,
            "includedPaths": [{ "path": "/*" }],
            "excludedPaths": excluded_paths,
            "vectorIndexes": vector_indexes,
        }),
        configs,
    })
}

/// An indexing-policy path for a storage name: `/name/*` when the name is a
/// plain identifier, `/"name"/*` when it needs quoting.
fn policy_path(storage_name: &str, suffix: &str) -> String {
    let plain = !storage_name.is_empty()
        && storage_name
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && storage_name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_');
    if plain {
        format!("/{storage_name}{suffix}")
    } else {
        format!("/{}{suffix}", json_string(storage_name))
    }
}

/// Normalize a policy path for comparison: Cosmos echoes a quoted segment
/// back verbatim, and `/"name"` and `/name` address the same property when
/// the name is a plain identifier. Comparing the raw strings would report a
/// container the service itself created from this policy as incompatible.
fn normalize_policy_path(path: &str) -> String {
    let Some(rest) = path.strip_prefix("/\"") else {
        return path.to_string();
    };
    // Find the closing quote, honouring backslash escapes.
    let bytes: Vec<char> = rest.chars().collect();
    let mut escaped = false;
    for (i, c) in bytes.iter().enumerate() {
        if *c == '"' && !escaped {
            let quoted: String = std::iter::once('"')
                .chain(bytes[..=i].iter().copied())
                .collect();
            let suffix: String = bytes[i + 1..].iter().collect();
            return match serde_json::from_str::<String>(&quoted) {
                Ok(segment) => format!("{}{suffix}", policy_path(&segment, "")),
                Err(_) => path.to_string(),
            };
        }
        escaped = *c == '\\' && !escaped;
    }
    path.to_string()
}

/// A JSON string literal for `value`, which is also a valid Cosmos SQL string
/// literal: both escape `"` and `\` the same way.
fn json_string(value: &str) -> String {
    Value::String(value.to_string()).to_string()
}

/// `c["name"]` — property access that works for any storage name, including
/// one that is not a bare identifier.
fn property_access(storage_name: &str) -> Result<String> {
    if storage_name.is_empty() || storage_name.contains('\0') {
        return Err(Error::Configuration(
            "an Azure Cosmos DB field storage name must be a non-empty string without NUL".into(),
        ));
    }
    Ok(format!("c[{}]", json_string(storage_name)))
}

// region: record validation

fn validate_key(value: &Value) -> Result<String> {
    let key = value.as_str().ok_or_else(|| {
        Error::Configuration("an Azure Cosmos DB item key must be a string".into())
    })?;
    if key.is_empty() || key.len() > ID_BYTE_LIMIT || key.contains(['/', '\\', '?', '#']) {
        return Err(Error::Configuration(format!(
            "an Azure Cosmos DB item key must be 1..={ID_BYTE_LIMIT} UTF-8 bytes and contain \
             none of '/', '\\\\', '?', '#'; got {key:?}"
        )));
    }
    Ok(key.to_string())
}

/// Reject a value Cosmos DB cannot store faithfully, naming the path to it.
///
/// Two of these are silent corruptions rather than service errors: an integer
/// beyond 2^53 comes back rounded, and a `f64` that is not finite is not JSON
/// at all (`serde_json` renders it as `null`), so the field would read as
/// null on the way out. Depth is a hard service limit.
fn validate_json(value: &Value, path: &str, depth: usize) -> Result<()> {
    if depth > MAX_JSON_DEPTH {
        return Err(Error::Configuration(format!(
            "{path} exceeds Azure Cosmos DB's maximum JSON nesting depth of {MAX_JSON_DEPTH}"
        )));
    }
    match value {
        Value::Null | Value::Bool(_) | Value::String(_) => Ok(()),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                if !(-MAX_SAFE_INTEGER..=MAX_SAFE_INTEGER).contains(&i) {
                    return Err(Error::Configuration(format!(
                        "{path} must fit exactly in an IEEE 754 binary64 JSON number"
                    )));
                }
                Ok(())
            } else if let Some(u) = n.as_u64() {
                if u > MAX_SAFE_INTEGER as u64 {
                    return Err(Error::Configuration(format!(
                        "{path} must fit exactly in an IEEE 754 binary64 JSON number"
                    )));
                }
                Ok(())
            } else if n.as_f64().is_some_and(f64::is_finite) {
                Ok(())
            } else {
                Err(Error::Configuration(format!(
                    "{path} must contain only finite JSON numbers"
                )))
            }
        }
        Value::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                validate_json(item, &format!("{path}[{i}]"), depth + 1)?;
            }
            Ok(())
        }
        Value::Object(map) => {
            for (k, item) in map {
                validate_json(item, &format!("{path}.{k}"), depth + 1)?;
            }
            Ok(())
        }
    }
}

/// Check a vector value's shape against its field: the right number of
/// elements, all finite, all inside the element type's range.
fn validate_vector(value: &Value, field: &VectorStoreField, config: &VectorConfig) -> Result<()> {
    let items = value.as_array().ok_or_else(|| {
        Error::Configuration(format!(
            "vector field '{}' must hold a dense array of numbers",
            field.name
        ))
    })?;
    if let Some(dimensions) = field.dimensions {
        if items.len() != dimensions {
            return Err(Error::Configuration(format!(
                "vector field '{}' requires exactly {dimensions} dimensions, got {}",
                field.name,
                items.len()
            )));
        }
    }
    let (min, max, integral) = match config.data_type {
        "int8" => (-128.0, 127.0, true),
        "uint8" => (0.0, 255.0, true),
        _ => (f64::from(f32::MIN), f64::from(f32::MAX), false),
    };
    for (i, item) in items.iter().enumerate() {
        if item.is_boolean() {
            return Err(Error::Configuration(format!(
                "vector field '{}' element {i} is a boolean, not a number",
                field.name
            )));
        }
        let n = item.as_f64().filter(|v| v.is_finite()).ok_or_else(|| {
            Error::Configuration(format!(
                "vector field '{}' element {i} must be a finite number",
                field.name
            ))
        })?;
        if integral && item.as_i64().is_none() {
            return Err(Error::Configuration(format!(
                "a {} vector requires integer elements, and field '{}' element {i} is not one",
                config.data_type, field.name
            )));
        }
        if n < min || n > max {
            return Err(Error::Configuration(format!(
                "a {} vector's elements must lie within [{min}, {max}], and field '{}' element \
                 {i} is {n}",
                config.data_type, field.name
            )));
        }
    }
    Ok(())
}

// region: the store

/// A connection to a Cosmos DB database, handing out vector collections.
///
/// Both authentication modes the rest of this crate offers are available:
/// [`CosmosVectorStore::new`] signs with a master key, and
/// [`CosmosVectorStore::with_token_credential`] uses any
/// [`TokenCredential`] — the only mode that works on an account with
/// `disableLocalAuth` set. As elsewhere in this crate, creating a database or
/// container is a control-plane operation that Entra ID RBAC cannot grant, so
/// under a credential the container must be provisioned out of band; the
/// error says so rather than surfacing a bare `403`.
pub struct CosmosVectorStore {
    client: Arc<CosmosRestClient>,
    database: String,
}

impl std::fmt::Debug for CosmosVectorStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CosmosVectorStore")
            .field("endpoint", &self.client.account_endpoint())
            .field("database", &self.database)
            .finish_non_exhaustive()
    }
}

impl CosmosVectorStore {
    /// Authenticate with the account's master/primary key.
    pub fn new(
        account_endpoint: impl Into<String>,
        key: impl Into<String>,
        database: impl Into<String>,
    ) -> Result<Self> {
        Ok(Self {
            client: Arc::new(
                CosmosRestClient::new(account_endpoint, key)?
                    .with_api_version(DEFAULT_VECTOR_API_VERSION),
            ),
            database: database.into(),
        })
    }

    /// Authenticate with Microsoft Entra ID.
    ///
    /// `scope` defaults to `https://cosmos.azure.com/.default`, or to
    /// `AZURE_COSMOS_AAD_SCOPE_OVERRIDE` when that is set.
    pub fn with_token_credential(
        account_endpoint: impl Into<String>,
        credential: Arc<dyn TokenCredential>,
        database: impl Into<String>,
        scope: Option<String>,
    ) -> Result<Self> {
        Ok(Self {
            client: Arc::new(
                CosmosRestClient::with_token_credential(account_endpoint, credential, scope)?
                    .with_api_version(DEFAULT_VECTOR_API_VERSION),
            ),
            database: database.into(),
        })
    }

    /// Pin a different `x-ms-version`. See [`DEFAULT_VECTOR_API_VERSION`].
    pub fn with_api_version(mut self, api_version: impl Into<String>) -> Result<Self> {
        let client = Arc::try_unwrap(self.client).map_err(|_| {
            Error::Configuration(
                "set the api-version before handing out any collection from this store".into(),
            )
        })?;
        self.client = Arc::new(client.with_api_version(api_version));
        Ok(self)
    }

    /// Create the database if it does not exist. Master-key auth only — see
    /// the type docs.
    pub async fn ensure_database_exists(&self) -> Result<()> {
        self.client
            .create_database_if_not_exists(&self.database)
            .await
    }

    /// Open one container as a typed collection.
    ///
    /// No I/O: the definition is validated against what Cosmos DB can express
    /// (the key must be stored as `id`, the distance function must be one
    /// Cosmos computes), but nothing is created or read. Call
    /// [`VectorCollection::ensure_collection_exists`] for that.
    pub fn collection(
        &self,
        name: impl Into<String>,
        definition: VectorStoreCollectionDefinition,
    ) -> Result<CosmosVectorCollection> {
        CosmosVectorCollection::new(
            Arc::clone(&self.client),
            self.database.clone(),
            name.into(),
            definition,
        )
    }
}

#[async_trait::async_trait]
impl VectorStore for CosmosVectorStore {
    fn get_collection(
        &self,
        name: &str,
        definition: VectorStoreCollectionDefinition,
    ) -> Result<Box<dyn VectorCollection>> {
        Ok(Box::new(self.collection(name, definition)?))
    }

    async fn list_collection_names(&self) -> Result<Vec<String>> {
        self.client.list_container_ids(&self.database).await
    }

    async fn collection_exists(&self, name: &str) -> Result<bool> {
        // A direct read beats the trait's default list-and-scan: one request,
        // and it stays correct in a database with more containers than one
        // page.
        Ok(self
            .client
            .read_container(&self.database, name)
            .await?
            .is_some())
    }
}

// region: the collection

/// One Cosmos DB container, as a [`VectorCollection`].
pub struct CosmosVectorCollection {
    client: Arc<CosmosRestClient>,
    database: String,
    container: String,
    definition: VectorStoreCollectionDefinition,
    configs: HashMap<String, VectorConfig>,
    vector_policy: Value,
    indexing_policy: Value,
}

impl std::fmt::Debug for CosmosVectorCollection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CosmosVectorCollection")
            .field("database", &self.database)
            .field("container", &self.container)
            .finish_non_exhaustive()
    }
}

impl CosmosVectorCollection {
    fn new(
        client: Arc<CosmosRestClient>,
        database: String,
        container: String,
        definition: VectorStoreCollectionDefinition,
    ) -> Result<Self> {
        let schema = prepare_schema(&definition)?;
        Ok(Self {
            client,
            database,
            container,
            definition,
            configs: schema.configs,
            vector_policy: schema.vector_policy,
            indexing_policy: schema.indexing_policy,
        })
    }

    /// The `POST /dbs/{db}/colls` body this collection would create, minus
    /// the `id` and `partitionKey` the client adds.
    ///
    /// Exposed for the same reason `AzureAISearchCollection::build_index` is:
    /// a caller provisioning the container out of band (ARM, Bicep, the CLI)
    /// can take the exact policies this connector validates against rather
    /// than hand-writing them and discovering the mismatch at first search.
    pub fn build_container_policies(&self) -> Map<String, Value> {
        let mut body = Map::new();
        body.insert("indexingPolicy".into(), self.indexing_policy.clone());
        body.insert("vectorEmbeddingPolicy".into(), self.vector_policy.clone());
        body
    }

    /// Check an existing container against this definition.
    ///
    /// A container whose vector policy disagrees with the definition cannot
    /// serve the searches this collection will issue — a missing embedding
    /// path makes `VectorDistance` a runtime error, and a *different*
    /// `distanceFunction` is worse: the query succeeds and ranks by a metric
    /// the caller did not ask for. Neither can be repaired by an update
    /// (Cosmos DB rejects a vector-policy change on an existing container),
    /// so this reports rather than fixes.
    fn validate_existing(&self, properties: &Value) -> Result<()> {
        let partition_paths = properties
            .get("partitionKey")
            .and_then(|p| p.get("paths"))
            .and_then(Value::as_array)
            .map(|paths| {
                paths
                    .iter()
                    .filter_map(Value::as_str)
                    .map(String::from)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if partition_paths != [VECTOR_PARTITION_KEY_PATH] {
            return Err(Error::Configuration(format!(
                "container '{}' partitions on {partition_paths:?}, but this connector addresses \
                 every item by its key in its own partition and so requires the single path \
                 '{VECTOR_PARTITION_KEY_PATH}'",
                self.container
            )));
        }

        let expected = embedding_set(&self.vector_policy)?;
        let actual = properties
            .get("vectorEmbeddingPolicy")
            .ok_or_else(|| {
                Error::Configuration(format!(
                    "container '{}' has no vector embedding policy; a vector collection's \
                     policy is fixed at creation, so it has to be created with one",
                    self.container
                ))
            })
            .and_then(embedding_set)?;
        if expected != actual {
            return Err(Error::Configuration(format!(
                "container '{}' has a vector embedding policy that disagrees with this \
                 collection definition (expected {expected:?}, found {actual:?}); a vector \
                 policy cannot be changed after creation",
                self.container
            )));
        }

        let indexing = properties.get("indexingPolicy").ok_or_else(|| {
            Error::Configuration(format!(
                "container '{}' has no indexing policy",
                self.container
            ))
        })?;
        let excluded: std::collections::BTreeSet<String> =
            policy_entries(indexing, "excludedPaths")
                .iter()
                .filter_map(|e| e.get("path").and_then(Value::as_str))
                .map(normalize_policy_path)
                .collect();
        for required in policy_entries(&self.indexing_policy, "excludedPaths") {
            let path = required
                .get("path")
                .and_then(Value::as_str)
                .map(normalize_policy_path)
                .unwrap_or_default();
            // Only the vector exclusions are load-bearing: indexing a vector
            // in the ordinary index is what a caller pays for in RUs, and the
            // service refuses to index one above 505 dimensions at all.
            if path.contains("/*") && path != "/_etag/*" && !excluded.contains(&path) {
                return Err(Error::Configuration(format!(
                    "container '{}' does not exclude '{path}' from its ordinary index, which a \
                     vector path must be",
                    self.container
                )));
            }
        }

        let expected_indexes = vector_index_map(&self.indexing_policy);
        let actual_indexes = vector_index_map(indexing);
        for (path, kind) in &expected_indexes {
            match actual_indexes.get(path) {
                None => {
                    return Err(Error::Configuration(format!(
                        "container '{}' has no vector index on '{path}'",
                        self.container
                    )))
                }
                Some(found) if !found.eq_ignore_ascii_case(kind) => {
                    return Err(Error::Configuration(format!(
                        "container '{}' indexes '{path}' as '{found}', but this collection \
                         declares '{kind}'",
                        self.container
                    )))
                }
                Some(_) => {}
            }
        }
        Ok(())
    }

    fn vector_config(
        &self,
        options: &VectorSearchOptions,
    ) -> Result<(&VectorStoreField, &VectorConfig)> {
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
        let config = self.configs.get(&field.name).ok_or_else(|| {
            Error::Configuration(format!("'{}' is not a vector field", field.name))
        })?;
        Ok((field, config))
    }

    /// Translate a portable filter into a Cosmos SQL predicate, appending its
    /// literals to `parameters` rather than inlining them.
    ///
    /// Exposed alongside [`Self::build_container_policies`] for a caller
    /// driving the REST API themselves. Returns `Ok(None)` when there is
    /// nothing to translate.
    pub fn prepare_filter(
        &self,
        options: &VectorSearchOptions,
        parameters: &mut Vec<(String, Value)>,
    ) -> Result<Option<String>> {
        let portable = match options.filter.as_ref() {
            Some(filter) => {
                filter.validate()?;
                Some(self.translate(filter, parameters)?)
            }
            None => None,
        };
        Ok(match (portable, options.provider_filter.as_deref()) {
            (None, None) => None,
            (Some(p), None) => Some(p),
            (None, Some(raw)) => Some(format!("({raw})")),
            // Conjoined rather than one silently winning: both were asked for.
            (Some(p), Some(raw)) => Some(format!("({p} AND ({raw}))")),
        })
    }

    fn translate(
        &self,
        expression: &FilterExpression,
        parameters: &mut Vec<(String, Value)>,
    ) -> Result<String> {
        match expression {
            FilterExpression::Group(group) => {
                let children = group
                    .filters
                    .iter()
                    .map(|child| self.translate(child, parameters))
                    .collect::<Result<Vec<_>>>()?;
                Ok(match group.operator {
                    FilterGroupOperator::Not => format!("(NOT {})", children[0]),
                    FilterGroupOperator::And => format!("({})", children.join(" AND ")),
                    FilterGroupOperator::Or => format!("({})", children.join(" OR ")),
                })
            }
            FilterExpression::Condition(filter) => self.translate_condition(filter, parameters),
        }
    }

    fn resolve_filter_field(&self, filter: &Filter) -> Result<&VectorStoreField> {
        if filter.field_name.contains('.') {
            return Err(Error::Configuration(format!(
                "nested portable filter paths are not supported here: '{}'",
                filter.field_name
            )));
        }
        let field = self
            .definition
            .try_get_field(&filter.field_name)
            .ok_or_else(|| {
                Error::Configuration(format!(
                    "filter field '{}' is not part of the collection definition",
                    filter.field_name
                ))
            })?;
        if field.field_type == FieldType::Vector {
            return Err(Error::Configuration(format!(
                "a vector field cannot be filtered on: '{}'",
                field.name
            )));
        }
        if field.field_type != FieldType::Key && field.is_indexed == Some(false) {
            return Err(Error::Configuration(format!(
                "filter field '{}' is excluded from this collection's index, so filtering on it \
                 would scan the container rather than use an index",
                field.name
            )));
        }
        Ok(field)
    }

    fn translate_condition(
        &self,
        filter: &Filter,
        parameters: &mut Vec<(String, Value)>,
    ) -> Result<String> {
        let field = self.resolve_filter_field(filter)?;
        let access = property_access(field.effective_storage_name())?;
        let declared = DeclaredType::of(field);
        let value = filter.value.as_ref();

        let scalar = |v: &Value| -> Result<()> {
            match v {
                Value::String(_) | Value::Bool(_) => Ok(()),
                Value::Number(_) => validate_json(v, "filter value", 0),
                _ => Err(Error::Configuration(
                    "an Azure Cosmos DB filter literal must be a string, boolean, or number".into(),
                )),
            }
        };
        let bind = |parameters: &mut Vec<(String, Value)>, v: Value| -> String {
            let name = format!("@filter_{}", parameters.len());
            parameters.push((name.clone(), v));
            name
        };
        let require_value = || -> Result<&Value> {
            value.ok_or_else(|| {
                Error::Configuration(format!(
                    "the '{}' filter operator requires a value",
                    filter.operator.as_str()
                ))
            })
        };

        match &filter.operator {
            FilterOperator::Exists => Ok(format!("IS_DEFINED({access})")),
            FilterOperator::IsNull => Ok(format!("(IS_DEFINED({access}) AND IS_NULL({access}))")),
            FilterOperator::IsNotNull => {
                Ok(format!("(IS_DEFINED({access}) AND NOT IS_NULL({access}))"))
            }
            op @ (FilterOperator::Eq | FilterOperator::Ne) => {
                let v = require_value()?;
                scalar(v)?;
                let is_eq = matches!(op, FilterOperator::Eq);
                if !declared.accepts(v) {
                    // The literal cannot equal any value this field can hold,
                    // so the comparison is decided statically. `ne` still
                    // requires presence: a missing field is a non-match for
                    // every operator but `exists`.
                    return Ok(if is_eq {
                        "false".to_string()
                    } else {
                        format!("IS_DEFINED({access})")
                    });
                }
                let p = bind(parameters, v.clone());
                Ok(if is_eq {
                    format!("(IS_DEFINED({access}) AND {access} = {p})")
                } else {
                    // A stored null is "present and not equal", which `!=`
                    // alone does not say in Cosmos SQL.
                    format!("(IS_DEFINED({access}) AND (IS_NULL({access}) OR {access} != {p}))")
                })
            }
            op @ (FilterOperator::Gt
            | FilterOperator::Gte
            | FilterOperator::Lt
            | FilterOperator::Lte
            | FilterOperator::Between) => {
                let v = require_value()?;
                let operands: Vec<Value> = if matches!(op, FilterOperator::Between) {
                    v.as_array()
                        .ok_or_else(|| {
                            Error::Configuration(
                                "the 'between' filter operator takes [lower, upper]".into(),
                            )
                        })?
                        .clone()
                } else {
                    vec![v.clone()]
                };
                let mut guards = vec![
                    format!("IS_DEFINED({access})"),
                    format!("NOT IS_NULL({access})"),
                ];
                for operand in &operands {
                    scalar(operand)?;
                    if !declared.accepts(operand) {
                        return Err(Error::Configuration(format!(
                            "an ordered comparison against field '{}' cannot use the literal \
                             {operand}, which is not of the field's declared type",
                            field.name
                        )));
                    }
                    if declared == DeclaredType::Unspecified {
                        // Cosmos SQL orders *across* types (number < string <
                        // array < object), so an untyped field would compare a
                        // string against a number and match. Upstream refuses
                        // an ordered filter on an untyped field outright;
                        // here the type hint is optional and usually unset, so
                        // the guard the refusal exists to provide is emitted
                        // instead.
                        let guard = DeclaredType::guard_for(operand).ok_or_else(|| {
                            Error::Configuration(
                                "an ordered comparison needs a string or numeric literal".into(),
                            )
                        })?;
                        guards.push(format!("{guard}({access})"));
                    }
                }
                if matches!(op, FilterOperator::Between) {
                    if operands.len() != 2 {
                        return Err(Error::Configuration(
                            "the 'between' filter operator takes exactly [lower, upper]".into(),
                        ));
                    }
                    let lower = bind(parameters, operands[0].clone());
                    let upper = bind(parameters, operands[1].clone());
                    guards.push(format!("{access} >= {lower}"));
                    guards.push(format!("{access} <= {upper}"));
                } else {
                    let sql = match op {
                        FilterOperator::Gt => ">",
                        FilterOperator::Gte => ">=",
                        FilterOperator::Lt => "<",
                        _ => "<=",
                    };
                    let p = bind(parameters, operands[0].clone());
                    guards.push(format!("{access} {sql} {p}"));
                }
                Ok(format!("({})", guards.join(" AND ")))
            }
            op @ (FilterOperator::In | FilterOperator::NotIn) => {
                let v = require_value()?;
                let items = v.as_array().ok_or_else(|| {
                    Error::Configuration(format!(
                        "the '{}' filter operator takes an array",
                        filter.operator.as_str()
                    ))
                })?;
                for item in items {
                    if !item.is_null() {
                        scalar(item)?;
                    }
                }
                let is_in = matches!(op, FilterOperator::In);
                if items.is_empty() {
                    return Ok(if is_in {
                        "false".to_string()
                    } else {
                        format!("(IS_DEFINED({access}) AND NOT IS_NULL({access}))")
                    });
                }
                let p = bind(parameters, Value::Array(items.clone()));
                let contained = format!("ARRAY_CONTAINS({p}, {access})");
                Ok(if is_in {
                    format!("(IS_DEFINED({access}) AND NOT IS_NULL({access}) AND {contained})")
                } else {
                    format!("(IS_DEFINED({access}) AND NOT IS_NULL({access}) AND NOT {contained})")
                })
            }
            op @ (FilterOperator::Contains
            | FilterOperator::ContainsAny
            | FilterOperator::ContainsAll) => {
                let v = require_value()?;
                if !matches!(declared, DeclaredType::List | DeclaredType::Unspecified) {
                    return Err(Error::Configuration(format!(
                        "a membership filter needs an array field, and '{}' declares a scalar \
                         type",
                        field.name
                    )));
                }
                let items: Vec<Value> = if matches!(op, FilterOperator::Contains) {
                    vec![v.clone()]
                } else {
                    v.as_array()
                        .ok_or_else(|| {
                            Error::Configuration(format!(
                                "the '{}' filter operator takes an array",
                                filter.operator.as_str()
                            ))
                        })?
                        .clone()
                };
                for item in &items {
                    if !item.is_null() {
                        scalar(item)?;
                    }
                }
                if items.is_empty() {
                    // `contains_all` of nothing holds for any array;
                    // `contains_any` of nothing holds for none.
                    return Ok(if matches!(op, FilterOperator::ContainsAll) {
                        format!("IS_ARRAY({access})")
                    } else {
                        "false".to_string()
                    });
                }
                let clauses: Vec<String> = items
                    .into_iter()
                    .map(|item| {
                        let p = bind(parameters, item);
                        format!("ARRAY_CONTAINS({access}, {p})")
                    })
                    .collect();
                let joiner = if matches!(op, FilterOperator::ContainsAll) {
                    " AND "
                } else {
                    " OR "
                };
                Ok(format!(
                    "(IS_ARRAY({access}) AND ({}))",
                    clauses.join(joiner)
                ))
            }
            op @ (FilterOperator::StartsWith
            | FilterOperator::EndsWith
            | FilterOperator::ContainsText) => {
                let v = require_value()?;
                if !matches!(declared, DeclaredType::Str | DeclaredType::Unspecified) {
                    return Err(Error::Configuration(format!(
                        "a string filter needs a string field, and '{}' declares another type",
                        field.name
                    )));
                }
                if !v.is_string() {
                    return Err(Error::Configuration(
                        "a string filter takes a string literal".into(),
                    ));
                }
                let function = match op {
                    FilterOperator::StartsWith => "STARTSWITH",
                    FilterOperator::EndsWith => "ENDSWITH",
                    _ => "CONTAINS",
                };
                let p = bind(parameters, v.clone());
                Ok(format!(
                    "(IS_STRING({access}) AND {function}({access}, {p}))"
                ))
            }
            FilterOperator::Provider(name) => Err(Error::Configuration(format!(
                "this connector defines no provider filter operators, so '{name}' cannot be \
                 translated; use `VectorSearchOptions::with_provider_filter` to pass a Cosmos SQL \
                 predicate verbatim"
            ))),
        }
    }

    /// The `{"name": c["name"], ...}` object projection for a result.
    fn projection(&self, include_vectors: bool) -> Result<String> {
        let names = self.definition.get_storage_names(include_vectors, true);
        let parts = names
            .iter()
            .map(|name| Ok(format!("{}: {}", json_string(name), property_access(name)?)))
            .collect::<Result<Vec<_>>>()?;
        Ok(format!("{{{}}}", parts.join(", ")))
    }
}

/// The comparable form of a vector-embedding policy: one tuple per embedding,
/// order-independent.
fn embedding_set(
    policy: &Value,
) -> Result<std::collections::BTreeSet<(String, String, String, i64)>> {
    let entries = policy
        .get("vectorEmbeddings")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            Error::Configuration(
                "a vector embedding policy needs a `vectorEmbeddings` array".into(),
            )
        })?;
    Ok(entries
        .iter()
        .map(|e| {
            (
                e.get("path")
                    .and_then(Value::as_str)
                    .map(normalize_policy_path)
                    .unwrap_or_default(),
                e.get("dataType")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_ascii_lowercase(),
                e.get("distanceFunction")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_ascii_lowercase(),
                e.get("dimensions").and_then(Value::as_i64).unwrap_or(-1),
            )
        })
        .collect())
}

fn policy_entries<'a>(policy: &'a Value, name: &str) -> Vec<&'a Value> {
    policy
        .get(name)
        .and_then(Value::as_array)
        .map(|entries| entries.iter().collect())
        .unwrap_or_default()
}

fn vector_index_map(policy: &Value) -> std::collections::BTreeMap<String, String> {
    policy_entries(policy, "vectorIndexes")
        .iter()
        .filter_map(|e| {
            let path = e.get("path").and_then(Value::as_str)?;
            let kind = e.get("type").and_then(Value::as_str).unwrap_or_default();
            Some((normalize_policy_path(path), kind.to_string()))
        })
        .collect()
}

#[async_trait::async_trait]
impl VectorCollection for CosmosVectorCollection {
    fn name(&self) -> &str {
        &self.container
    }

    fn definition(&self) -> &VectorStoreCollectionDefinition {
        &self.definition
    }

    async fn ensure_collection_exists(&self) -> Result<()> {
        if let Some(properties) = self
            .client
            .read_container(&self.database, &self.container)
            .await?
        {
            return self.validate_existing(&properties);
        }
        self.client
            .create_container_with_body(
                &self.database,
                &self.container,
                VECTOR_PARTITION_KEY_PATH,
                &self.build_container_policies(),
            )
            .await?;
        // Read back rather than trusting the create: a `409` was tolerated
        // above, so the container that now exists may be someone else's, with
        // a policy this collection cannot search.
        match self
            .client
            .read_container(&self.database, &self.container)
            .await?
        {
            Some(properties) => self.validate_existing(&properties),
            None => Err(Error::service(format!(
                "container '{}' was created but cannot be read back",
                self.container
            ))),
        }
    }

    async fn collection_exists(&self) -> Result<bool> {
        Ok(self
            .client
            .read_container(&self.database, &self.container)
            .await?
            .is_some())
    }

    async fn ensure_collection_deleted(&self) -> Result<()> {
        self.client
            .delete_container(&self.database, &self.container)
            .await
    }

    async fn upsert(&self, records: Vec<Value>) -> Result<Vec<Value>> {
        let total = records.len();
        let mut keys = Vec::with_capacity(total);
        for (index, record) in records.into_iter().enumerate() {
            let stored = self.definition.to_storage(&record)?;
            let object = stored.as_object().ok_or_else(|| {
                Error::Configuration("a vector store record must be a JSON object".into())
            })?;
            // Every declared field must be present: a partial record would
            // silently *replace* the stored one with fewer fields, because an
            // upsert is a whole-document write.
            for field in self.definition.fields() {
                let storage_name = field.effective_storage_name();
                let Some(value) = object.get(storage_name) else {
                    return Err(Error::Configuration(format!(
                        "record at index {index} is missing field '{}'; an Azure Cosmos DB upsert \
                         replaces the whole document, so a partial record would drop the fields \
                         it omits",
                        field.name
                    )));
                };
                match field.field_type {
                    FieldType::Vector => {
                        if !value.is_null() {
                            let config = self.configs.get(&field.name).ok_or_else(|| {
                                Error::Configuration(format!("'{}' is not a vector", field.name))
                            })?;
                            validate_vector(value, field, config)?;
                        }
                    }
                    _ => {
                        if !value.is_null() && !DeclaredType::of(field).accepts(value) {
                            return Err(Error::Configuration(format!(
                                "record at index {index} field '{}' does not hold its declared \
                                 type '{}'",
                                field.name,
                                field.type_.as_deref().unwrap_or("")
                            )));
                        }
                    }
                }
                validate_json(value, &format!("record[{index}].{storage_name}"), 0)?;
            }
            let key = validate_key(object.get("id").unwrap_or(&Value::Null))?;
            let encoded = serde_json::to_vec(&stored)?;
            if encoded.len() > ITEM_SIZE_LIMIT {
                return Err(Error::Configuration(format!(
                    "record at index {index} serializes to {} bytes, above Azure Cosmos DB's 2 \
                     MiB item limit",
                    encoded.len()
                )));
            }
            self.client
                .upsert_document(&self.database, &self.container, &key, &stored)
                .await
                .map_err(|e| {
                    // Each record is its own request, so a mid-batch failure
                    // leaves the earlier ones written. Saying which index
                    // failed makes the retry exact rather than a re-run of the
                    // whole batch.
                    Error::service(format!(
                        "Azure Cosmos DB upsert completed {index}/{total} records before failing \
                         on input index {index}: {e}. The records before it are written; retry \
                         from there with the same keys."
                    ))
                })?;
            keys.push(Value::String(key));
        }
        Ok(keys)
    }

    async fn get(&self, keys: Vec<Value>, include_vectors: bool) -> Result<Vec<Option<Value>>> {
        let mut out = Vec::with_capacity(keys.len());
        for key in keys {
            let key = validate_key(&key)?;
            // A point read: the partition key is the id, so this never fans
            // out.
            let found = self
                .client
                .get_document(&self.database, &self.container, &key, &key)
                .await?;
            out.push(match found {
                Some(document) => Some(self.definition.from_storage(&document, include_vectors)?),
                None => None,
            });
        }
        Ok(out)
    }

    async fn delete(&self, keys: Vec<Value>) -> Result<()> {
        for key in keys {
            let key = validate_key(&key)?;
            self.client
                .delete_document(&self.database, &self.container, &key, &key)
                .await?;
        }
        Ok(())
    }

    async fn search(
        &self,
        vector: Vec<f32>,
        options: &VectorSearchOptions,
    ) -> Result<Vec<VectorSearchResult>> {
        options.validate()?;
        let (field, config) = self.vector_config(options)?;
        if let Some(dimensions) = field.dimensions {
            if vector.len() != dimensions {
                return Err(Error::Configuration(format!(
                    "query vector has {} dimensions but field '{}' declares {dimensions}",
                    vector.len(),
                    field.name
                )));
            }
        }
        if let Some(bad) = vector.iter().position(|v| !v.is_finite()) {
            return Err(Error::Configuration(format!(
                "query vector element {bad} is not finite"
            )));
        }

        let mut parameters: Vec<(String, Value)> = vec![(
            "@vector".to_string(),
            Value::Array(
                vector
                    .iter()
                    .map(|v| json!(f64::from(*v)))
                    .collect::<Vec<_>>(),
            ),
        )];
        let where_clause = self.prepare_filter(options, &mut parameters)?;

        let distance = format!(
            "VectorDistance({}, @vector)",
            property_access(&config.storage_name)?
        );
        let projection = self.projection(options.include_vectors)?;
        // `skip` is applied to the returned page rather than pushed into the
        // query: Cosmos DB does not accept `OFFSET` alongside an `ORDER BY
        // VectorDistance`, and the limit has to cover both halves anyway —
        // asking for `top` and then dropping `skip` of them would return
        // fewer than `top` records.
        let limit = options.skip.saturating_add(options.top);
        parameters.push(("@top".to_string(), json!(limit)));
        let mut query = format!(
            "SELECT TOP @top VALUE {{\"record\": {projection}, \"score\": {distance}}} FROM c"
        );
        if let Some(clause) = &where_clause {
            query.push_str(&format!(" WHERE {clause}"));
        }
        // No ASC/DESC: Cosmos DB orders a `VectorDistance` expression by
        // closeness for the metric the container declares, which is the
        // opposite direction for `euclidean` than for the two similarities.
        query.push_str(&format!(" ORDER BY {distance}"));

        let parameter_refs: Vec<(&str, Value)> = parameters
            .iter()
            .map(|(name, value)| (name.as_str(), value.clone()))
            .collect();
        let rows = self
            .client
            .query_documents_cross_partition(
                &self.database,
                &self.container,
                &query,
                &parameter_refs,
            )
            .await?;

        rows.into_iter()
            .skip(options.skip)
            .map(|row| {
                let record = row.get("record").ok_or_else(|| {
                    Error::service(
                        "Azure Cosmos DB search result is missing its record".to_string(),
                    )
                })?;
                Ok(VectorSearchResult {
                    record: self
                        .definition
                        .from_storage(record, options.include_vectors)?,
                    score: row.get("score").and_then(Value::as_f64),
                    // The score is in the collection's own declared distance
                    // function — see `cosmos_distance` — so the definition
                    // answers `higher_is_closer` correctly and there is
                    // nothing to override.
                    score_kind: None,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn definition() -> VectorStoreCollectionDefinition {
        VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id").with_type("str"),
            VectorStoreField::data("text").with_type("str"),
            VectorStoreField::data("rank").with_type("int"),
            VectorStoreField::data("tags").with_type("list"),
            VectorStoreField::data("loose"),
            VectorStoreField::vector("embedding", 3),
        ])
        .expect("valid definition")
    }

    fn collection() -> CosmosVectorCollection {
        let client = Arc::new(
            CosmosRestClient::new("https://acct.documents.azure.com:443/", test_key())
                .expect("client"),
        );
        CosmosVectorCollection::new(client, "db".into(), "coll".into(), definition())
            .expect("collection")
    }

    /// A synthetic, deterministic base64 key, built at runtime so no
    /// high-entropy literal appears in the source.
    fn test_key() -> String {
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.encode((0u8..64).collect::<Vec<u8>>())
    }

    fn translate(filter: FilterExpression) -> Result<(String, Vec<(String, Value)>)> {
        let collection = collection();
        let options = VectorSearchOptions::new(5).with_filter(filter);
        let mut parameters = Vec::new();
        let clause = collection.prepare_filter(&options, &mut parameters)?;
        Ok((clause.unwrap_or_default(), parameters))
    }

    // region: schema

    #[test]
    fn the_policies_declare_the_vector_and_exclude_it_from_the_ordinary_index() {
        let collection = collection();
        let policies = collection.build_container_policies();
        assert_eq!(
            policies["vectorEmbeddingPolicy"]["vectorEmbeddings"][0],
            json!({
                "path": "/embedding",
                "dataType": "float32",
                "distanceFunction": "cosine",
                "dimensions": 3,
            })
        );
        assert_eq!(
            policies["indexingPolicy"]["vectorIndexes"][0],
            json!({ "path": "/embedding", "type": "quantizedFlat" })
        );
        let excluded = policies["indexingPolicy"]["excludedPaths"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["path"].as_str().unwrap().to_string())
            .collect::<Vec<_>>();
        assert!(excluded.contains(&"/embedding/*".to_string()));
        assert!(excluded.contains(&"/_etag/?".to_string()));
    }

    #[test]
    fn a_non_indexed_data_field_is_excluded_from_the_index() {
        let definition = VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id").with_type("str"),
            VectorStoreField::data("blob").with_type("str"),
            VectorStoreField::vector("embedding", 3),
        ])
        .unwrap();
        let mut fields = definition.fields().to_vec();
        fields[1].is_indexed = Some(false);
        let definition = VectorStoreCollectionDefinition::new(fields).unwrap();
        let client = Arc::new(
            CosmosRestClient::new("https://acct.documents.azure.com:443/", test_key()).unwrap(),
        );
        let collection =
            CosmosVectorCollection::new(client, "db".into(), "coll".into(), definition).unwrap();
        let excluded = collection.build_container_policies()["indexingPolicy"]["excludedPaths"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["path"].as_str().unwrap().to_string())
            .collect::<Vec<_>>();
        assert!(excluded.contains(&"/blob/*".to_string()));
    }

    #[test]
    fn a_key_not_stored_as_id_is_refused() {
        let definition = VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("key").with_type("str"),
            VectorStoreField::vector("embedding", 3),
        ])
        .unwrap();
        let err = prepare_schema(&definition).unwrap_err().to_string();
        assert!(err.contains("stored as 'id'"), "{err}");
    }

    #[test]
    fn a_definition_with_no_vector_field_is_refused() {
        let definition = VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id").with_type("str"),
            VectorStoreField::data("text").with_type("str"),
        ])
        .unwrap();
        let err = prepare_schema(&definition).unwrap_err().to_string();
        assert!(err.contains("at least one vector field"), "{err}");
    }

    #[test]
    fn cosine_distance_is_refused_and_names_cosine_similarity() {
        // Cosmos computes cosine as a *similarity*; accepting the distance
        // spelling would hand back a score whose direction the caller's own
        // `higher_is_closer` reads backwards.
        let field = VectorStoreField::vector("embedding", 3)
            .with_distance_function(DistanceFunction::new(DistanceFunction::COSINE_DISTANCE));
        let err = cosmos_distance(&field).unwrap_err().to_string();
        assert!(err.contains("cosine_similarity"), "{err}");
    }

    #[test]
    fn each_supported_distance_and_index_kind_maps() {
        for (declared, expected) in [
            (DistanceFunction::COSINE_SIMILARITY, "cosine"),
            (DistanceFunction::DOT_PROD, "dotproduct"),
            (DistanceFunction::EUCLIDEAN_DISTANCE, "euclidean"),
        ] {
            let field = VectorStoreField::vector("v", 3)
                .with_distance_function(DistanceFunction::new(declared));
            assert_eq!(cosmos_distance(&field).unwrap(), expected);
        }
        for (declared, expected) in [
            (IndexKind::DEFAULT, "quantizedFlat"),
            (IndexKind::QUANTIZED_FLAT, "quantizedFlat"),
            (IndexKind::FLAT, "flat"),
            (IndexKind::DISK_ANN, "diskANN"),
        ] {
            let field = VectorStoreField::vector("v", 3).with_index_kind(IndexKind::new(declared));
            assert_eq!(cosmos_index_kind(&field).unwrap(), expected);
        }
        let hnsw =
            VectorStoreField::vector("v", 3).with_index_kind(IndexKind::new(IndexKind::HNSW));
        assert!(cosmos_index_kind(&hnsw).is_err());
    }

    #[test]
    fn a_flat_index_above_505_dimensions_is_refused() {
        let field =
            VectorStoreField::vector("v", 506).with_index_kind(IndexKind::new(IndexKind::FLAT));
        let err = prepare_vector_config(&field).unwrap_err().to_string();
        assert!(err.contains("505"), "{err}");
        // The same size is fine on the default index.
        assert!(prepare_vector_config(&VectorStoreField::vector("v", 506)).is_ok());
    }

    // region: filters

    #[test]
    fn eq_requires_presence_so_a_missing_field_is_a_non_match() {
        let (clause, parameters) = translate(Filter::eq("text", "hello").unwrap().into()).unwrap();
        assert_eq!(
            clause,
            r#"(IS_DEFINED(c["text"]) AND c["text"] = @filter_0)"#
        );
        assert_eq!(parameters, vec![("@filter_0".to_string(), json!("hello"))]);
    }

    #[test]
    fn ne_is_not_the_negation_of_eq() {
        // A missing field matches neither, and a stored null matches `ne`.
        let (clause, _) = translate(Filter::ne("text", "hello").unwrap().into()).unwrap();
        assert_eq!(
            clause,
            r#"(IS_DEFINED(c["text"]) AND (IS_NULL(c["text"]) OR c["text"] != @filter_0))"#
        );
    }

    #[test]
    fn an_incompatible_literal_decides_eq_and_ne_statically() {
        // `rank` is an int; no stored value can equal "x", and every present
        // one differs from it.
        let (eq, parameters) = translate(Filter::eq("rank", "x").unwrap().into()).unwrap();
        assert_eq!(eq, "false");
        assert!(parameters.is_empty());
        let (ne, _) = translate(Filter::ne("rank", "x").unwrap().into()).unwrap();
        assert_eq!(ne, r#"IS_DEFINED(c["rank"])"#);
    }

    #[test]
    fn an_ordered_filter_on_an_untyped_field_carries_a_type_guard() {
        // Cosmos SQL orders across types, so without the guard `loose > 5`
        // would also match every string stored in that field.
        let (clause, _) = translate(Filter::gt("loose", 5).unwrap().into()).unwrap();
        assert!(clause.contains(r#"IS_NUMBER(c["loose"])"#), "{clause}");
        let (text, _) = translate(Filter::gt("loose", "m").unwrap().into()).unwrap();
        assert!(text.contains(r#"IS_STRING(c["loose"])"#), "{text}");
        // A declared field needs no guard: the type is already pinned.
        let (typed, _) = translate(Filter::gt("rank", 5).unwrap().into()).unwrap();
        assert!(!typed.contains("IS_NUMBER"), "{typed}");
    }

    #[test]
    fn an_ordered_filter_against_the_wrong_declared_type_is_refused() {
        let err = translate(Filter::gt("rank", "x").unwrap().into())
            .unwrap_err()
            .to_string();
        assert!(err.contains("declared type"), "{err}");
    }

    #[test]
    fn between_binds_both_bounds_inclusively() {
        let (clause, parameters) =
            translate(Filter::between("rank", 1, 10).unwrap().into()).unwrap();
        assert!(clause.contains(r#"c["rank"] >= @filter_0"#), "{clause}");
        assert!(clause.contains(r#"c["rank"] <= @filter_1"#), "{clause}");
        assert_eq!(parameters.len(), 2);
    }

    #[test]
    fn empty_in_and_not_in_decide_without_a_parameter() {
        let (in_clause, parameters) = translate(
            Filter::new("text", FilterOperator::In, Some(json!([])))
                .unwrap()
                .into(),
        )
        .unwrap();
        assert_eq!(in_clause, "false");
        assert!(parameters.is_empty());
        let (not_in, _) = translate(
            Filter::new("text", FilterOperator::NotIn, Some(json!([])))
                .unwrap()
                .into(),
        )
        .unwrap();
        assert_eq!(
            not_in,
            r#"(IS_DEFINED(c["text"]) AND NOT IS_NULL(c["text"]))"#
        );
    }

    #[test]
    fn membership_filters_need_an_array_field() {
        let (clause, _) = translate(Filter::contains("tags", "a").unwrap().into()).unwrap();
        assert_eq!(
            clause,
            r#"(IS_ARRAY(c["tags"]) AND (ARRAY_CONTAINS(c["tags"], @filter_0)))"#
        );
        let err = translate(Filter::contains("text", "a").unwrap().into())
            .unwrap_err()
            .to_string();
        assert!(err.contains("array field"), "{err}");
    }

    #[test]
    fn contains_all_joins_with_and_and_contains_any_with_or() {
        let (all, _) = translate(
            Filter::new("tags", FilterOperator::ContainsAll, Some(json!(["a", "b"])))
                .unwrap()
                .into(),
        )
        .unwrap();
        assert!(all.contains(" AND ARRAY_CONTAINS"), "{all}");
        let (any, _) = translate(
            Filter::new("tags", FilterOperator::ContainsAny, Some(json!(["a", "b"])))
                .unwrap()
                .into(),
        )
        .unwrap();
        assert!(any.contains(" OR ARRAY_CONTAINS"), "{any}");
    }

    #[test]
    fn string_filters_guard_on_is_string() {
        let (clause, _) = translate(Filter::starts_with("text", "he").unwrap().into()).unwrap();
        assert_eq!(
            clause,
            r#"(IS_STRING(c["text"]) AND STARTSWITH(c["text"], @filter_0))"#
        );
    }

    #[test]
    fn presence_operators_are_the_only_ones_a_missing_field_can_satisfy() {
        let (exists, _) = translate(
            Filter::new("text", FilterOperator::Exists, None)
                .unwrap()
                .into(),
        )
        .unwrap();
        assert_eq!(exists, r#"IS_DEFINED(c["text"])"#);
        let (is_null, _) = translate(
            Filter::new("text", FilterOperator::IsNull, None)
                .unwrap()
                .into(),
        )
        .unwrap();
        assert_eq!(is_null, r#"(IS_DEFINED(c["text"]) AND IS_NULL(c["text"]))"#);
    }

    #[test]
    fn a_filter_literal_cannot_escape_into_the_query_text() {
        // Every literal is bound, so a value carrying a quote is a value.
        let (clause, parameters) =
            translate(Filter::eq("text", "a\" OR 1=1 --").unwrap().into()).unwrap();
        assert!(!clause.contains("OR 1=1"), "{clause}");
        assert_eq!(parameters[0].1, json!("a\" OR 1=1 --"));
    }

    #[test]
    fn a_vector_field_cannot_be_filtered_on() {
        let err = translate(Filter::eq("embedding", 1).unwrap().into())
            .unwrap_err()
            .to_string();
        assert!(err.contains("vector field"), "{err}");
    }

    #[test]
    fn a_provider_operator_is_refused_and_names_the_escape_hatch() {
        let filter = Filter::new(
            "text",
            FilterOperator::provider("azure_cosmos.match").unwrap(),
            Some(json!("x")),
        )
        .unwrap();
        let err = translate(filter.into()).unwrap_err().to_string();
        assert!(err.contains("with_provider_filter"), "{err}");
    }

    #[test]
    fn a_portable_filter_and_a_provider_filter_are_conjoined() {
        let collection = collection();
        let options = VectorSearchOptions::new(5)
            .with_filter(Filter::eq("text", "hello").unwrap())
            .with_provider_filter("IS_DEFINED(c[\"extra\"])");
        let mut parameters = Vec::new();
        let clause = collection
            .prepare_filter(&options, &mut parameters)
            .unwrap()
            .unwrap();
        assert!(clause.contains("AND (IS_DEFINED("), "{clause}");
    }

    #[test]
    fn groups_render_with_their_operator() {
        let group = agent_framework_core::vectors::FilterGroup::and(vec![
            Filter::eq("text", "a").unwrap().into(),
            Filter::eq("rank", 1).unwrap().into(),
        ])
        .unwrap();
        let (clause, parameters) = translate(group).unwrap();
        assert!(clause.starts_with("((IS_DEFINED"), "{clause}");
        assert!(clause.contains(" AND "), "{clause}");
        assert_eq!(parameters.len(), 2);
    }

    // region: records

    #[test]
    fn a_key_with_a_path_separator_is_refused() {
        for bad in ["a/b", "a\\b", "a?b", "a#b", ""] {
            assert!(
                validate_key(&json!(bad)).is_err(),
                "expected {bad:?} to be refused"
            );
        }
        assert_eq!(validate_key(&json!("a-b_c")).unwrap(), "a-b_c");
    }

    #[test]
    fn an_integer_beyond_binary64_is_refused_rather_than_silently_rounded() {
        let err = validate_json(&json!(9_007_199_254_740_993i64), "record[0].n", 0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("binary64"), "{err}");
        assert!(validate_json(&json!(9_007_199_254_740_991i64), "x", 0).is_ok());
    }

    #[test]
    fn a_vector_of_the_wrong_length_or_with_a_non_finite_element_is_refused() {
        let field = VectorStoreField::vector("embedding", 3);
        let config = prepare_vector_config(&field).unwrap();
        assert!(validate_vector(&json!([1.0, 2.0]), &field, &config).is_err());
        assert!(validate_vector(&json!([1.0, 2.0, true]), &field, &config).is_err());
        assert!(validate_vector(&json!([1.0, 2.0, 3.0]), &field, &config).is_ok());
    }

    #[test]
    fn an_int8_vector_requires_integer_elements_in_range() {
        let field = VectorStoreField::vector("embedding", 2).with_type("int8");
        let config = prepare_vector_config(&field).unwrap();
        assert!(validate_vector(&json!([1.5, 2.0]), &field, &config).is_err());
        assert!(validate_vector(&json!([200, 2]), &field, &config).is_err());
        assert!(validate_vector(&json!([-128, 127]), &field, &config).is_ok());
    }

    // region: policy comparison

    #[test]
    fn a_container_whose_distance_function_differs_is_refused() {
        let collection = collection();
        let mut properties = json!({
            "partitionKey": { "paths": ["/id"], "kind": "Hash" },
            "indexingPolicy": collection.indexing_policy.clone(),
        });
        properties["vectorEmbeddingPolicy"] = json!({
            "vectorEmbeddings": [{
                "path": "/embedding",
                "dataType": "float32",
                "distanceFunction": "euclidean",
                "dimensions": 3,
            }]
        });
        let err = collection
            .validate_existing(&properties)
            .unwrap_err()
            .to_string();
        assert!(err.contains("vector embedding policy"), "{err}");
    }

    #[test]
    fn a_container_created_from_these_policies_validates() {
        let collection = collection();
        let properties = json!({
            "partitionKey": { "paths": ["/id"], "kind": "Hash" },
            "indexingPolicy": collection.indexing_policy.clone(),
            "vectorEmbeddingPolicy": collection.vector_policy.clone(),
        });
        collection
            .validate_existing(&properties)
            .expect("compatible");
    }

    #[test]
    fn a_container_on_another_partition_key_is_refused() {
        let collection = collection();
        let properties = json!({
            "partitionKey": { "paths": ["/tenant"], "kind": "Hash" },
            "indexingPolicy": collection.indexing_policy.clone(),
            "vectorEmbeddingPolicy": collection.vector_policy.clone(),
        });
        let err = collection
            .validate_existing(&properties)
            .unwrap_err()
            .to_string();
        assert!(err.contains("/id"), "{err}");
    }

    #[test]
    fn a_quoted_policy_path_compares_equal_to_its_plain_form() {
        assert_eq!(normalize_policy_path("/\"embedding\"/*"), "/embedding/*");
        assert_eq!(normalize_policy_path("/embedding/*"), "/embedding/*");
        // A name that genuinely needs quoting keeps them.
        assert_eq!(normalize_policy_path("/\"a b\"/*"), "/\"a b\"/*");
    }
}
