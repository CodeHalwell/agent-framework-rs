//! An Azure AI Search [`VectorStore`]: index management, document CRUD, and
//! vector / keyword-hybrid retrieval.
//!
//! Ports upstream's `agent_framework_azure_ai_search._vector_store` (#8153).
//! Where that module drives the `azure-search-documents` SDK, this one speaks
//! the Search REST API directly over `reqwest` — the same choice the context
//! provider in this crate already made — so the two share an endpoint, an
//! api-version, and the [`SearchAuth`](crate::AzureAISearchProvider) shapes.
//!
//! ```no_run
//! use agent_framework_azure_ai_search::AzureAISearchStore;
//! use agent_framework_core::vectors::{
//!     Filter, VectorSearchOptions, VectorStore, VectorStoreCollectionDefinition, VectorStoreField,
//! };
//!
//! # async fn demo() -> agent_framework_core::error::Result<()> {
//! let store = AzureAISearchStore::with_api_key("https://my-search.search.windows.net", "admin-key");
//! let definition = VectorStoreCollectionDefinition::new(vec![
//!     VectorStoreField::key("id"),
//!     VectorStoreField::data("text").full_text_indexed(),
//!     VectorStoreField::data("year").with_type("int").indexed(),
//!     VectorStoreField::vector("embedding", 1536),
//! ])?;
//! let docs = store.get_collection("docs", definition)?;
//! docs.ensure_collection_exists().await?;
//!
//! let hits = docs
//!     .search(
//!         vec![0.0; 1536],
//!         &VectorSearchOptions::new(5).with_filter(Filter::gte("year", 2020)?),
//!     )
//!     .await?;
//! # let _ = hits;
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use agent_framework_azure::TokenCredential;
use agent_framework_core::error::{Error, Result};
use agent_framework_core::vectors::{
    DistanceFunction, FieldType, Filter, FilterExpression, FilterGroupOperator, FilterOperator,
    IndexKind, VectorCollection, VectorSearchOptions, VectorSearchResult, VectorStore,
    VectorStoreCollectionDefinition, VectorStoreField,
};
use async_trait::async_trait;
use serde_json::{json, Map, Value};

use crate::{SearchAuth, DEFAULT_API_VERSION, SEARCH_SCOPE};

/// The largest `skip + top` Azure AI Search will serve from a vector query.
const MAX_RESULT_WINDOW: usize = 10_000;

/// The largest number of document actions Azure AI Search accepts in one
/// indexing batch. Upserts and deletes are chunked to this.
const MAX_BATCH_ACTIONS: usize = 1_000;

/// A connection to an Azure AI Search service.
///
/// Hands out [`AzureAISearchCollection`]s (one per index) and manages the
/// indexes and aliases themselves.
#[derive(Clone)]
pub struct AzureAISearchStore {
    http: reqwest::Client,
    endpoint: String,
    api_version: String,
    scope: String,
    auth: SearchAuth,
}

impl std::fmt::Debug for AzureAISearchStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AzureAISearchStore")
            .field("endpoint", &self.endpoint)
            .field("api_version", &self.api_version)
            .field("auth", &self.auth.kind())
            .finish_non_exhaustive()
    }
}

impl AzureAISearchStore {
    fn build(endpoint: String, auth: SearchAuth) -> Self {
        Self {
            http: reqwest::Client::new(),
            endpoint: endpoint.trim_end_matches('/').to_string(),
            api_version: DEFAULT_API_VERSION.to_string(),
            scope: SEARCH_SCOPE.to_string(),
            auth,
        }
    }

    /// Connect with a search **admin key** (`api-key` header).
    ///
    /// Index creation and deletion need an admin key; a query key can read
    /// and search but not manage indexes.
    pub fn with_api_key(endpoint: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self::build(endpoint.into(), SearchAuth::ApiKey(api_key.into()))
    }

    /// Connect with a [`TokenCredential`] (`Authorization: Bearer`, scope
    /// [`SEARCH_SCOPE`]).
    pub fn with_token_credential(
        endpoint: impl Into<String>,
        credential: Arc<dyn TokenCredential>,
    ) -> Self {
        Self::build(endpoint.into(), SearchAuth::Credential(credential))
    }

    /// Override the REST API version (default [`DEFAULT_API_VERSION`]).
    pub fn with_api_version(mut self, api_version: impl Into<String>) -> Self {
        self.api_version = api_version.into();
        self
    }

    /// Override the Entra ID token scope (default [`SEARCH_SCOPE`]).
    pub fn with_scope(mut self, scope: impl Into<String>) -> Self {
        self.scope = scope.into();
        self
    }

    fn url(&self, path: &str) -> String {
        format!(
            "{}/{path}{}api-version={}",
            self.endpoint,
            if path.contains('?') { "&" } else { "?" },
            self.api_version
        )
    }

    async fn send(&self, req: reqwest::RequestBuilder) -> Result<reqwest::Response> {
        let req = match &self.auth {
            SearchAuth::ApiKey(key) => req.header("api-key", key),
            SearchAuth::Credential(cred) => {
                req.bearer_auth(cred.get_token_for_scope(&self.scope).await?)
            }
        };
        req.send()
            .await
            .map_err(|e| Error::service(format!("Azure AI Search request failed: {e}")))
    }

    /// Open a collection as the concrete [`AzureAISearchCollection`].
    ///
    /// [`VectorStore::get_collection`] returns a `Box<dyn VectorCollection>`,
    /// which is what makes providers swappable but also hides everything the
    /// trait does not declare — [`AzureAISearchCollection::search_hybrid`] and
    /// [`AzureAISearchCollection::build_index`] among them. Use this when the
    /// caller has already committed to Azure AI Search.
    pub fn collection(
        &self,
        name: &str,
        definition: VectorStoreCollectionDefinition,
    ) -> Result<AzureAISearchCollection> {
        AzureAISearchCollection::new(self.clone(), name.to_string(), definition)
    }

    /// Point an index alias at `collection_name`, creating it if needed.
    ///
    /// An alias lets callers keep querying one name while the index behind it
    /// is rebuilt and swapped.
    pub async fn create_or_update_alias(
        &self,
        alias_name: &str,
        collection_name: &str,
    ) -> Result<()> {
        let body = json!({ "name": alias_name, "indexes": [collection_name] });
        let resp = self
            .send(
                self.http
                    .put(self.url(&format!("aliases('{}')", escape_path(alias_name))))
                    .json(&body),
            )
            .await?;
        expect_success(resp, "create or update alias")
            .await
            .map(drop)
    }

    /// Delete an index alias. Deleting an absent alias is not an error.
    pub async fn delete_alias(&self, alias_name: &str) -> Result<()> {
        let resp = self
            .send(
                self.http
                    .delete(self.url(&format!("aliases('{}')", escape_path(alias_name)))),
            )
            .await?;
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(());
        }
        expect_success(resp, "delete alias").await.map(drop)
    }
}

#[async_trait]
impl VectorStore for AzureAISearchStore {
    fn get_collection(
        &self,
        name: &str,
        definition: VectorStoreCollectionDefinition,
    ) -> Result<Box<dyn VectorCollection>> {
        Ok(Box::new(self.collection(name, definition)?))
    }

    async fn list_collection_names(&self) -> Result<Vec<String>> {
        let resp = self
            .send(self.http.get(self.url("indexes?$select=name")))
            .await?;
        let body = expect_json(expect_success(resp, "list indexes").await?).await?;
        Ok(body
            .get("value")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.get("name").and_then(Value::as_str))
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default())
    }

    async fn collection_exists(&self, name: &str) -> Result<bool> {
        // A direct GET, rather than the trait's default full listing: the
        // service answers it with one lookup, and listing needs a permission
        // that reading one index does not.
        let resp = self
            .send(
                self.http
                    .get(self.url(&format!("indexes('{}')", escape_path(name)))),
            )
            .await?;
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(false);
        }
        expect_success(resp, "get index").await.map(|_| true)
    }
}

/// One Azure AI Search index, as a [`VectorCollection`].
pub struct AzureAISearchCollection {
    store: AzureAISearchStore,
    name: String,
    definition: VectorStoreCollectionDefinition,
    /// Storage name -> EDM type, resolved once at construction. The filter
    /// translator needs it to reject a comparison the service would refuse,
    /// and the index builder writes it.
    field_types: std::collections::HashMap<String, String>,
}

impl std::fmt::Debug for AzureAISearchCollection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AzureAISearchCollection")
            .field("name", &self.name)
            .field("endpoint", &self.store.endpoint)
            .finish_non_exhaustive()
    }
}

impl AzureAISearchCollection {
    fn new(
        store: AzureAISearchStore,
        name: String,
        definition: VectorStoreCollectionDefinition,
    ) -> Result<Self> {
        let mut field_types = std::collections::HashMap::new();
        for field in definition.fields() {
            let storage_name = prepare_field_name(field.effective_storage_name())?;
            field_types.insert(storage_name, edm_type(field)?);
        }
        Ok(Self {
            store,
            name,
            definition,
            field_types,
        })
    }

    fn index_url(&self, suffix: &str) -> String {
        self.store
            .url(&format!("indexes('{}'){suffix}", escape_path(&self.name)))
    }

    /// The index schema this collection's definition describes.
    ///
    /// Exposed so a caller can inspect or extend the schema — add a semantic
    /// configuration, a scoring profile, an analyzer — and create the index
    /// themselves, rather than being limited to what
    /// [`VectorCollection::ensure_collection_exists`] generates.
    pub fn build_index(&self) -> Result<Value> {
        let mut fields = Vec::new();
        let mut algorithms = Vec::new();
        let mut profiles = Vec::new();

        for field in self.definition.fields() {
            let name = prepare_field_name(field.effective_storage_name())?;
            let type_ = edm_type(field)?;
            let mut spec = Map::new();
            spec.insert("name".into(), json!(name));
            spec.insert("type".into(), json!(type_));
            spec.insert("key".into(), json!(field.field_type == FieldType::Key));

            match field.field_type {
                FieldType::Vector => {
                    // Azure rejects a filterable, sortable, facetable or
                    // analyzed vector field outright; the definition's
                    // `is_indexed` means "filterable", which is not what a
                    // vector field is indexed *for*.
                    if field.is_indexed == Some(true) {
                        return Err(Error::Configuration(format!(
                            "Azure AI Search vector field '{}' cannot be filterable",
                            field.name
                        )));
                    }
                    let dimensions = field.dimensions.ok_or_else(|| {
                        Error::Configuration(format!(
                            "vector field '{}' declares no dimensions",
                            field.name
                        ))
                    })?;
                    let profile = format!("{name}_profile");
                    let algorithm = format!("{name}_algorithm");
                    spec.insert("searchable".into(), json!(true));
                    spec.insert("filterable".into(), json!(false));
                    spec.insert("vectorSearchDimensions".into(), json!(dimensions));
                    spec.insert("vectorSearchProfileName".into(), json!(profile));

                    let metric = search_metric(field)?;
                    let kind = match field
                        .index_kind
                        .as_ref()
                        .map(IndexKind::as_str)
                        .unwrap_or(IndexKind::DEFAULT)
                    {
                        IndexKind::FLAT => "exhaustiveKnn",
                        IndexKind::HNSW | IndexKind::DEFAULT => "hnsw",
                        other => {
                            return Err(Error::Configuration(format!(
                                "Azure AI Search supports the 'hnsw' and 'flat' index kinds, not \
                                 '{other}'"
                            )))
                        }
                    };
                    let parameters_key = if kind == "hnsw" {
                        "hnswParameters"
                    } else {
                        "exhaustiveKnnParameters"
                    };
                    algorithms.push(json!({
                        "name": algorithm,
                        "kind": kind,
                        parameters_key: { "metric": metric },
                    }));
                    profiles.push(json!({
                        "name": profile,
                        "algorithm": algorithm,
                    }));
                }
                FieldType::Key => {
                    // A key is always filterable: `get` reads it back, and
                    // `delete` addresses records by it.
                    spec.insert("filterable".into(), json!(true));
                    spec.insert("searchable".into(), json!(false));
                }
                FieldType::Data => {
                    spec.insert("filterable".into(), json!(field.is_indexed == Some(true)));
                    spec.insert(
                        "searchable".into(),
                        json!(field.is_full_text_indexed == Some(true)),
                    );
                }
            }
            fields.push(Value::Object(spec));
        }

        let mut index = Map::new();
        index.insert("name".into(), json!(self.name));
        index.insert("fields".into(), Value::Array(fields));
        if !algorithms.is_empty() {
            index.insert(
                "vectorSearch".into(),
                json!({ "algorithms": algorithms, "profiles": profiles }),
            );
        }
        Ok(Value::Object(index))
    }

    /// Translate a portable filter into an OData `$filter` expression.
    ///
    /// Exposed for the same reason [`Self::build_index`] is: a caller driving
    /// the REST API themselves can reuse the translation rather than
    /// hand-writing OData. Returns `Ok(None)` when there is nothing to
    /// translate.
    pub fn prepare_filter(&self, options: &VectorSearchOptions) -> Result<Option<String>> {
        let portable = match options.filter.as_ref() {
            Some(filter) => {
                filter.validate()?;
                Some(self.translate(filter)?)
            }
            None => None,
        };
        Ok(match (portable, options.provider_filter.as_deref()) {
            (None, None) => None,
            (Some(p), None) => Some(p),
            (None, Some(raw)) => Some(raw.to_string()),
            // Conjoined rather than one silently winning: both were asked for.
            (Some(p), Some(raw)) => Some(format!("({p}) and ({raw})")),
        })
    }

    fn translate(&self, expression: &FilterExpression) -> Result<String> {
        match expression {
            FilterExpression::Group(group) => {
                let children = group
                    .filters
                    .iter()
                    .map(|child| self.translate(child))
                    .collect::<Result<Vec<_>>>()?;
                Ok(match group.operator {
                    FilterGroupOperator::Not => format!("not ({})", children[0]),
                    FilterGroupOperator::And => format!("({})", children.join(" and ")),
                    FilterGroupOperator::Or => format!("({})", children.join(" or ")),
                })
            }
            FilterExpression::Condition(filter) => self.translate_condition(filter),
        }
    }

    fn translate_condition(&self, filter: &Filter) -> Result<String> {
        let field = self
            .definition
            .try_get_field(&filter.field_name)
            .ok_or_else(|| {
                Error::Configuration(format!(
                    "filter field '{}' is not part of the collection definition (Azure AI Search \
                     does not support nested portable field paths)",
                    filter.field_name
                ))
            })?;
        let name = prepare_field_name(field.effective_storage_name())?;
        let value = filter.value.as_ref();

        if let FilterOperator::Provider(op) = &filter.operator {
            if op == "azure_ai_search.match" {
                let text = value.and_then(Value::as_str).ok_or_else(|| {
                    Error::Configuration(
                        "azure_ai_search.match requires a string query".to_string(),
                    )
                })?;
                if field.is_full_text_indexed != Some(true) {
                    return Err(Error::Configuration(format!(
                        "azure_ai_search.match requires a full-text-indexed field, and '{}' is \
                         not one",
                        field.name
                    )));
                }
                return Ok(format!(
                    "search.ismatch({}, {}, 'simple', 'any')",
                    odata_literal(&json!(text))?,
                    odata_literal(&json!(name))?
                ));
            }
            return Err(Error::Configuration(format!(
                "Azure AI Search does not understand the provider filter operator '{op}'"
            )));
        }

        if field.field_type == FieldType::Vector
            || (field.field_type != FieldType::Key && field.is_indexed != Some(true))
        {
            return Err(Error::Configuration(format!(
                "field '{}' must be filterable (declare it with `.indexed()`) to filter on it",
                field.name
            )));
        }

        let op = filter.operator.as_str();
        // Three refusals, each because the OData translation would be subtly
        // wrong rather than merely unsupported.
        if matches!(
            filter.operator,
            FilterOperator::Exists | FilterOperator::IsNull | FilterOperator::IsNotNull
        ) {
            return Err(Error::Configuration(format!(
                "portable '{op}' cannot preserve missing-versus-null semantics in Azure AI \
                 Search, which does not distinguish an absent field from a null one"
            )));
        }
        if filter.operator == FilterOperator::Ne {
            return Err(Error::Configuration(
                "portable 'ne' cannot preserve missing-versus-null semantics in Azure AI Search; \
                 use a NOT group, whose missing-field semantics differ from 'ne'"
                    .into(),
            ));
        }
        if matches!(
            filter.operator,
            FilterOperator::StartsWith | FilterOperator::EndsWith | FilterOperator::ContainsText
        ) {
            return Err(Error::Configuration(format!(
                "Azure AI Search cannot implement a literal '{op}'; use the \
                 `azure_ai_search.match` operator for tokenized full-text search"
            )));
        }

        let type_ = self.field_types.get(&name).cloned().unwrap_or_default();
        if type_ == "Edm.DateTimeOffset" {
            return Err(Error::Configuration(
                "portable date comparisons are not supported by AzureAISearchCollection".into(),
            ));
        }

        if matches!(
            filter.operator,
            FilterOperator::Contains | FilterOperator::ContainsAny | FilterOperator::ContainsAll
        ) {
            if type_ != "Collection(Edm.String)" {
                return Err(Error::Configuration(format!(
                    "portable collection membership supports only string collection fields, and \
                     '{}' is {type_}",
                    field.name
                )));
            }
            let owned;
            let values: &[Value] = if filter.operator == FilterOperator::Contains {
                owned = vec![value.cloned().unwrap_or(Value::Null)];
                &owned
            } else {
                value
                    .and_then(Value::as_array)
                    .map(Vec::as_slice)
                    .ok_or_else(|| {
                        Error::Configuration(format!("'{op}' requires an array value"))
                    })?
            };
            if values.is_empty() {
                if filter.operator == FilterOperator::ContainsAll {
                    return Err(Error::Configuration(
                        "an empty 'contains_all' cannot distinguish an absent collection from an \
                         empty one in Azure AI Search"
                            .into(),
                    ));
                }
                return Ok("false".into());
            }
            let clauses = values
                .iter()
                .map(|item| {
                    Ok(format!(
                        "{name}/any(v: v eq {})",
                        odata_literal_typed(item, "Edm.String")?
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            let joiner = if filter.operator == FilterOperator::ContainsAll {
                " and "
            } else {
                " or "
            };
            return Ok(format!("({})", clauses.join(joiner)));
        }

        if type_.starts_with("Collection(") {
            return Err(Error::Configuration(format!(
                "Azure AI Search collection field '{}' supports membership, not scalar \
                 comparisons",
                field.name
            )));
        }

        match filter.operator {
            FilterOperator::Eq
            | FilterOperator::Gt
            | FilterOperator::Gte
            | FilterOperator::Lt
            | FilterOperator::Lte => {
                let odata_op = match filter.operator {
                    FilterOperator::Eq => "eq",
                    FilterOperator::Gt => "gt",
                    FilterOperator::Gte => "ge",
                    FilterOperator::Lt => "lt",
                    _ => "le",
                };
                condition(&name, odata_op, value.unwrap_or(&Value::Null), &type_)
            }
            FilterOperator::Between => {
                let bounds = value
                    .and_then(Value::as_array)
                    .ok_or_else(|| Error::Configuration("'between' requires two bounds".into()))?;
                Ok(format!(
                    "({} and {})",
                    condition(&name, "ge", &bounds[0], &type_)?,
                    condition(&name, "le", &bounds[1], &type_)?
                ))
            }
            FilterOperator::In | FilterOperator::NotIn => {
                let items = value.and_then(Value::as_array).ok_or_else(|| {
                    Error::Configuration(format!("'{op}' requires an array value"))
                })?;
                let comparisons = items
                    .iter()
                    .map(|item| condition(&name, "eq", item, &type_))
                    .collect::<Result<Vec<_>>>()?;
                let contained = if comparisons.is_empty() {
                    "false".to_string()
                } else {
                    format!("({})", comparisons.join(" or "))
                };
                Ok(if filter.operator == FilterOperator::In {
                    contained
                } else {
                    // `ne null` keeps a missing field out of the negation:
                    // without it every record lacking the field would match.
                    format!("({name} ne null and not ({contained}))")
                })
            }
            _ => Err(Error::Configuration(format!(
                "Azure AI Search does not support the filter operator '{op}'"
            ))),
        }
    }

    /// The `$select` projection: every field the caller expects back.
    fn projection(&self, include_vectors: bool) -> Vec<String> {
        self.definition
            .fields()
            .iter()
            .filter(|f| include_vectors || f.field_type != FieldType::Vector)
            .filter_map(|f| prepare_field_name(f.effective_storage_name()).ok())
            .collect()
    }

    /// Build the `docs/search` body shared by vector and keyword-hybrid
    /// search.
    fn search_body(
        &self,
        vector: &[f32],
        options: &VectorSearchOptions,
        text: Option<&str>,
    ) -> Result<Value> {
        options.validate()?;
        if options.skip.saturating_add(options.top) > MAX_RESULT_WINDOW {
            return Err(Error::Configuration(format!(
                "Azure AI Search cannot serve a vector result window beyond \
                 {MAX_RESULT_WINDOW} records (skip {} + top {})",
                options.skip, options.top
            )));
        }
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
        if let Some(bad) = vector.iter().position(|v| !v.is_finite()) {
            return Err(Error::Configuration(format!(
                "query vector element {bad} is not finite; Azure AI Search rejects a non-finite \
                 vector"
            )));
        }
        if let Some(dimensions) = field.dimensions {
            if vector.len() != dimensions {
                return Err(Error::Configuration(format!(
                    "query vector has {} dimensions but field '{}' declares {dimensions}",
                    vector.len(),
                    field.name
                )));
            }
        }

        let mut body = Map::new();
        // `k` has to cover the whole window: asking for `top` neighbors and
        // then skipping into them returns fewer records than requested.
        let k = options.top.saturating_add(options.skip).max(1);
        body.insert(
            "vectorQueries".into(),
            json!([{
                "kind": "vector",
                "vector": vector,
                "fields": prepare_field_name(field.effective_storage_name())?,
                "k": k,
            }]),
        );
        // Filter before the vector search rather than after it: a post-filter
        // discards matches from an already-truncated neighbor list, so a
        // selective filter returns far fewer than `top` records.
        body.insert("vectorFilterMode".into(), json!("preFilter"));
        body.insert("top".into(), json!(options.top));
        if options.skip > 0 {
            body.insert("skip".into(), json!(options.skip));
        }
        body.insert(
            "select".into(),
            json!(self.projection(options.include_vectors).join(",")),
        );
        if let Some(filter) = self.prepare_filter(options)? {
            body.insert("filter".into(), json!(filter));
        }
        if let Some(text) = text {
            body.insert("search".into(), json!(text));
            let search_fields: Vec<String> = self
                .definition
                .fields()
                .iter()
                .filter(|f| f.field_type == FieldType::Data && f.is_full_text_indexed == Some(true))
                .filter_map(|f| prepare_field_name(f.effective_storage_name()).ok())
                .collect();
            if search_fields.is_empty() {
                return Err(Error::Configuration(
                    "keyword-hybrid search needs a full-text-indexed data field; declare one with \
                     `.full_text_indexed()`"
                        .into(),
                ));
            }
            body.insert("searchFields".into(), json!(search_fields.join(",")));
        }
        Ok(Value::Object(body))
    }

    async fn run_search(&self, body: Value) -> Result<Vec<VectorSearchResult>> {
        let resp = self
            .store
            .send(
                self.store
                    .http
                    .post(self.index_url("/docs/search"))
                    .json(&body),
            )
            .await?;
        let payload = expect_json(expect_success(resp, "search").await?).await?;
        let mut out = Vec::new();
        for hit in payload
            .get("value")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
        {
            let score = hit.get("@search.score").and_then(Value::as_f64);
            // Strip the service's `@search.*` annotations before mapping back:
            // they are not fields of the record, and `from_storage` would
            // carry them into it.
            let stored = Value::Object(
                hit.as_object()
                    .map(|o| {
                        o.iter()
                            .filter(|(k, _)| !k.starts_with("@search."))
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect()
                    })
                    .unwrap_or_default(),
            );
            out.push(VectorSearchResult {
                record: self.definition.from_storage(&stored, true)?,
                score,
            });
        }
        Ok(out)
    }

    /// Keyword-hybrid search: a vector query and a full-text query, fused by
    /// the service with Reciprocal Rank Fusion.
    ///
    /// This is Azure AI Search's headline retrieval mode and the reason the
    /// context provider in this crate exists, but the [`VectorCollection`]
    /// trait's `search` takes only a vector, so it lives here as an inherent
    /// method. The returned `score` is then an RRF score, which is not
    /// comparable to the `@search.score` a pure vector query returns.
    pub async fn search_hybrid(
        &self,
        text: &str,
        vector: Vec<f32>,
        options: &VectorSearchOptions,
    ) -> Result<Vec<VectorSearchResult>> {
        let body = self.search_body(&vector, options, Some(text))?;
        self.run_search(body).await
    }

    /// Post an indexing batch, chunked to what the service accepts, and fail
    /// on any per-document error the batch reports.
    async fn index_documents(&self, actions: Vec<Value>) -> Result<()> {
        for chunk in actions.chunks(MAX_BATCH_ACTIONS) {
            let resp = self
                .store
                .send(
                    self.store
                        .http
                        .post(self.index_url("/docs/index"))
                        .json(&json!({ "value": chunk })),
                )
                .await?;
            // 207 is a *partial* success: some documents were rejected. It is
            // not an error status, so a plain `is_success()` check would
            // report a silently incomplete write as a complete one.
            let status = resp.status();
            let payload = expect_json(expect_success(resp, "index documents").await?).await?;
            let failures: Vec<String> = payload
                .get("value")
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter(|item| item.get("status").and_then(Value::as_bool) == Some(false))
                        .map(|item| {
                            format!(
                                "{}: {}",
                                item.get("key").and_then(Value::as_str).unwrap_or("?"),
                                item.get("errorMessage")
                                    .and_then(Value::as_str)
                                    .unwrap_or("unknown error")
                            )
                        })
                        .collect()
                })
                .unwrap_or_default();
            if !failures.is_empty() {
                return Err(Error::service(format!(
                    "Azure AI Search rejected {} of {} documents (HTTP {status}): {}",
                    failures.len(),
                    chunk.len(),
                    failures.join("; ")
                )));
            }
        }
        Ok(())
    }

    /// The key, as the string Azure AI Search addresses documents by.
    fn key_string(&self, key: &Value) -> Result<String> {
        match key {
            Value::String(s) => Ok(s.clone()),
            // An Azure AI Search key is always `Edm.String`; a caller holding
            // numeric keys would otherwise get a 404 per document with no
            // indication why.
            other => Err(Error::Configuration(format!(
                "Azure AI Search keys must be strings, not {other}"
            ))),
        }
    }
}

#[async_trait]
impl VectorCollection for AzureAISearchCollection {
    fn name(&self) -> &str {
        &self.name
    }

    fn definition(&self) -> &VectorStoreCollectionDefinition {
        &self.definition
    }

    async fn ensure_collection_exists(&self) -> Result<()> {
        if self.store.collection_exists(&self.name).await? {
            return Ok(());
        }
        let index = self.build_index()?;
        let resp = self
            .store
            .send(self.store.http.put(self.index_url("")).json(&index))
            .await?;
        expect_success(resp, "create index").await.map(drop)
    }

    async fn collection_exists(&self) -> Result<bool> {
        self.store.collection_exists(&self.name).await
    }

    async fn ensure_collection_deleted(&self) -> Result<()> {
        let resp = self
            .store
            .send(self.store.http.delete(self.index_url("")))
            .await?;
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(());
        }
        expect_success(resp, "delete index").await.map(drop)
    }

    async fn upsert(&self, records: Vec<Value>) -> Result<Vec<Value>> {
        let key_name = self.definition.key_field().name.clone();
        let mut keys = Vec::with_capacity(records.len());
        let mut actions = Vec::with_capacity(records.len());
        for record in &records {
            let key = record.get(&key_name).cloned().ok_or_else(|| {
                Error::Configuration(format!("record is missing its key field '{key_name}'"))
            })?;
            self.key_string(&key)?;
            keys.push(key);
            let mut stored = self.definition.to_storage(record)?;
            // `upload`, not `mergeOrUpload`: the trait contract is insert-or-
            // **replace**, and `InMemoryVectorStore` replaces wholesale. Merge
            // semantics keep a field the new record omits, so a record whose
            // optional field was cleared would keep its old value here and
            // nowhere else — still matching filters and still coming back in
            // search results, which is a divergence a caller would only find
            // in production.
            stored
                .as_object_mut()
                .ok_or_else(|| Error::Configuration("record is not a JSON object".into()))?
                .insert("@search.action".into(), json!("upload"));
            actions.push(stored);
        }
        if actions.is_empty() {
            return Ok(keys);
        }
        self.index_documents(actions).await?;
        Ok(keys)
    }

    async fn get(&self, keys: Vec<Value>, include_vectors: bool) -> Result<Vec<Option<Value>>> {
        let select = self.projection(include_vectors).join(",");
        let mut out = Vec::with_capacity(keys.len());
        for key in &keys {
            let key = self.key_string(key)?;
            let url = self.store.url(&format!(
                "indexes('{}')/docs('{}')?$select={}",
                escape_path(&self.name),
                escape_path(&key),
                urlencode(&select)
            ));
            let resp = self.store.send(self.store.http.get(url)).await?;
            if resp.status() == reqwest::StatusCode::NOT_FOUND {
                // A missing key yields `None` in its slot rather than
                // shortening the result, as the trait documents.
                out.push(None);
                continue;
            }
            let doc = expect_json(expect_success(resp, "get document").await?).await?;
            out.push(Some(self.definition.from_storage(&doc, include_vectors)?));
        }
        Ok(out)
    }

    async fn delete(&self, keys: Vec<Value>) -> Result<()> {
        if keys.is_empty() {
            return Ok(());
        }
        let key_storage_name =
            prepare_field_name(self.definition.key_field().effective_storage_name())?;
        let actions = keys
            .iter()
            .map(|key| {
                Ok(json!({
                    "@search.action": "delete",
                    key_storage_name.clone(): self.key_string(key)?,
                }))
            })
            .collect::<Result<Vec<_>>>()?;
        self.index_documents(actions).await
    }

    async fn search(
        &self,
        vector: Vec<f32>,
        options: &VectorSearchOptions,
    ) -> Result<Vec<VectorSearchResult>> {
        let body = self.search_body(&vector, options, None)?;
        self.run_search(body).await
    }
}

/// A field's EDM type: the explicit `type_` hint when it names one, otherwise
/// inferred from the field's role.
///
/// Upstream infers from a Python annotation (`str`, `int`, `list[str]`, ...).
/// Here `type_` is a free-form hint, so both spellings are accepted: an EDM
/// type verbatim (`Edm.Int32`), or one of the same short names upstream maps.
fn edm_type(field: &VectorStoreField) -> Result<String> {
    if field.field_type == FieldType::Vector {
        // The one dense numeric collection every Search tier supports. The
        // narrower packings (`Edm.Half`, `Edm.Int16`, `Edm.SByte`) need a
        // matching client-side encoding this port does not do.
        return Ok("Collection(Edm.Single)".into());
    }
    let hint = field.type_.as_deref().unwrap_or("str");
    let scalar = |name: &str| -> Option<&'static str> {
        Some(match name {
            "str" | "string" => "Edm.String",
            "int" => "Edm.Int64",
            "int32" => "Edm.Int32",
            "float" => "Edm.Double",
            "bool" => "Edm.Boolean",
            "datetime" => "Edm.DateTimeOffset",
            _ => return None,
        })
    };
    let type_ = if hint.starts_with("Edm.") || hint.starts_with("Collection(Edm.") {
        hint.to_string()
    } else if let Some(edm) = scalar(hint) {
        edm.to_string()
    } else if let Some(inner) = hint
        .strip_prefix("list[")
        .and_then(|rest| rest.strip_suffix(']'))
        .and_then(scalar)
    {
        format!("Collection({inner})")
    } else {
        return Err(Error::Configuration(format!(
            "Azure AI Search cannot infer a field type from '{hint}'; use an explicit EDM type \
             such as 'Edm.String'"
        )));
    };
    if field.field_type == FieldType::Key && type_ != "Edm.String" {
        return Err(Error::Configuration(format!(
            "Azure AI Search keys must be Edm.String, and '{}' declares {type_}",
            field.name
        )));
    }
    Ok(type_)
}

/// The Azure metric name for a field's distance function.
fn search_metric(field: &VectorStoreField) -> Result<&'static str> {
    match field
        .distance_function
        .as_ref()
        .map(DistanceFunction::as_str)
        .unwrap_or(DistanceFunction::COSINE_SIMILARITY)
    {
        DistanceFunction::COSINE_SIMILARITY | DistanceFunction::COSINE_DISTANCE => Ok("cosine"),
        DistanceFunction::DOT_PROD => Ok("dotProduct"),
        DistanceFunction::EUCLIDEAN_DISTANCE => Ok("euclidean"),
        // Not "fall back to cosine": a collection asking for a metric the
        // service does not have would then be indexed under a different one
        // and rank confidently wrong.
        other => Err(Error::Configuration(format!(
            "Azure AI Search does not support the distance function '{other}'"
        ))),
    }
}

/// Azure AI Search field names are simple ASCII identifiers.
fn prepare_field_name(name: &str) -> Result<String> {
    let mut chars = name.chars();
    let ok = matches!(chars.next(), Some(c) if c.is_ascii_alphabetic())
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !ok {
        return Err(Error::Configuration(format!(
            "Azure AI Search field storage names must be simple ASCII identifiers: '{name}'"
        )));
    }
    Ok(name.to_string())
}

/// One `name op literal` comparison, type-checked against the field's EDM
/// type.
fn condition(name: &str, op: &str, value: &Value, type_: &str) -> Result<String> {
    if value.is_null() {
        return Err(Error::Configuration(
            "Azure AI Search cannot distinguish a missing field from an explicit null".into(),
        ));
    }
    let compatible = match type_ {
        "Edm.String" => value.is_string(),
        "Edm.Boolean" => value.is_boolean(),
        "Edm.Int32" | "Edm.Int64" | "Edm.Double" => value.is_number(),
        // An unknown type is not silently trusted.
        _ => false,
    };
    if !compatible {
        // An `eq` against an incompatible type cannot match anything, so it
        // is answered as `false` rather than refused — that is a filter that
        // is merely over-specific, not one that is malformed. Every other
        // operator would be asking the service to order two different types.
        if op == "eq" && !value.is_null() && !value.is_array() && !value.is_object() {
            return Ok("false".into());
        }
        return Err(Error::Configuration(format!(
            "filter value {value} is incompatible with the Azure AI Search type {type_} of \
             '{name}'"
        )));
    }
    if op != "eq" && matches!(type_, "Edm.String" | "Edm.Boolean") {
        return Err(Error::Configuration(format!(
            "Azure AI Search supports portable ordered comparisons only on numeric fields, and \
             '{name}' is {type_}"
        )));
    }
    Ok(format!(
        "{name} {op} {}",
        odata_literal_typed(value, type_)?
    ))
}

/// Render a value as an OData literal.
fn odata_literal(value: &Value) -> Result<String> {
    match value {
        // OData escapes a single quote by doubling it. Getting this wrong is
        // how a filter value becomes an injection into the expression.
        Value::String(s) => Ok(format!("'{}'", s.replace('\'', "''"))),
        Value::Bool(b) => Ok(if *b { "true".into() } else { "false".into() }),
        Value::Number(n) => {
            let f = n.as_f64().unwrap_or(f64::NAN);
            if !f.is_finite() {
                return Err(Error::Configuration(
                    "Azure AI Search filter numbers must be finite".into(),
                ));
            }
            Ok(n.to_string())
        }
        other => Err(Error::Configuration(format!(
            "Azure AI Search filters take string, boolean, or finite numeric values, not {other}"
        ))),
    }
}

/// [`odata_literal`], first checking the value against an expected EDM type.
fn odata_literal_typed(value: &Value, type_: &str) -> Result<String> {
    if type_ == "Edm.String" && !value.is_string() {
        return Err(Error::Configuration(format!(
            "expected a string filter value for an Edm.String field, got {value}"
        )));
    }
    odata_literal(value)
}

/// Escape a value going into an OData key path segment (`indexes('name')`).
fn escape_path(value: &str) -> String {
    value.replace('\'', "''")
}

/// Percent-encode a query-string value.
fn urlencode(value: &str) -> String {
    value
        .bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b',' => {
                (b as char).to_string()
            }
            other => format!("%{other:02X}"),
        })
        .collect()
}

/// Turn a non-success response into an [`Error::ServiceStatus`], carrying the
/// service's own message.
async fn expect_success(resp: reqwest::Response, what: &str) -> Result<reqwest::Response> {
    let status = resp.status();
    if status.is_success() {
        return Ok(resp);
    }
    // Carried through so `RetryingChatClient`-style backoff honors the
    // service's own pacing on a 429 rather than computing its own.
    let retry_after = resp
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<f64>().ok());
    let body = resp.text().await.unwrap_or_default();
    Err(Error::service_status(
        status.as_u16(),
        format!("Azure AI Search {what} failed ({status}): {body}"),
        retry_after,
    ))
}

async fn expect_json(resp: reqwest::Response) -> Result<Value> {
    resp.json()
        .await
        .map_err(|e| Error::service(format!("invalid Azure AI Search response json: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::vectors::FilterGroup;

    fn definition() -> VectorStoreCollectionDefinition {
        VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id"),
            VectorStoreField::data("text").full_text_indexed().indexed(),
            VectorStoreField::data("year").with_type("int").indexed(),
            VectorStoreField::data("rating")
                .with_type("float")
                .indexed(),
            VectorStoreField::data("tags")
                .with_type("list[str]")
                .indexed(),
            VectorStoreField::data("note"),
            VectorStoreField::vector("embedding", 3),
        ])
        .unwrap()
    }

    fn collection() -> AzureAISearchCollection {
        AzureAISearchCollection::new(
            AzureAISearchStore::with_api_key("https://s.search.windows.net", "k"),
            "docs".into(),
            definition(),
        )
        .unwrap()
    }

    fn filter_of(filter: Filter) -> Result<String> {
        let options = VectorSearchOptions::new(3).with_filter(filter);
        Ok(collection().prepare_filter(&options)?.unwrap())
    }

    // region: index schema

    #[test]
    fn build_index_maps_fields_and_vector_profiles() {
        let index = collection().build_index().unwrap();
        let fields = index["fields"].as_array().unwrap();
        let by_name = |name: &str| {
            fields
                .iter()
                .find(|f| f["name"] == name)
                .unwrap_or_else(|| panic!("no field {name}"))
                .clone()
        };

        assert_eq!(by_name("id")["type"], "Edm.String");
        assert_eq!(by_name("id")["key"], true);
        assert_eq!(by_name("id")["filterable"], true);
        assert_eq!(by_name("text")["searchable"], true);
        assert_eq!(by_name("year")["type"], "Edm.Int64");
        assert_eq!(by_name("tags")["type"], "Collection(Edm.String)");
        // Not declared `.indexed()`, so it is not filterable — the mapping a
        // caller relies on when the translator refuses to filter on it.
        assert_eq!(by_name("note")["filterable"], false);

        let vector = by_name("embedding");
        assert_eq!(vector["type"], "Collection(Edm.Single)");
        assert_eq!(vector["vectorSearchDimensions"], 3);
        assert_eq!(vector["vectorSearchProfileName"], "embedding_profile");
        let search = &index["vectorSearch"];
        assert_eq!(search["algorithms"][0]["kind"], "hnsw");
        assert_eq!(
            search["algorithms"][0]["hnswParameters"]["metric"],
            "cosine"
        );
        assert_eq!(search["profiles"][0]["name"], "embedding_profile");
    }

    #[test]
    fn a_flat_index_kind_becomes_exhaustive_knn() {
        let definition = VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id"),
            VectorStoreField::vector("v", 2)
                .with_index_kind(IndexKind::new(IndexKind::FLAT))
                .with_distance_function(DistanceFunction::new(DistanceFunction::DOT_PROD)),
        ])
        .unwrap();
        let c = AzureAISearchCollection::new(
            AzureAISearchStore::with_api_key("https://s.search.windows.net", "k"),
            "docs".into(),
            definition,
        )
        .unwrap();
        let index = c.build_index().unwrap();
        assert_eq!(
            index["vectorSearch"]["algorithms"][0]["kind"],
            "exhaustiveKnn"
        );
        assert_eq!(
            index["vectorSearch"]["algorithms"][0]["exhaustiveKnnParameters"]["metric"],
            "dotProduct"
        );
    }

    #[test]
    fn an_unsupported_metric_is_refused_rather_than_defaulted() {
        // Falling back to cosine would index the collection under a metric it
        // did not ask for and rank confidently wrong.
        let definition = VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id"),
            VectorStoreField::vector("v", 2)
                .with_distance_function(DistanceFunction::new(DistanceFunction::MANHATTAN)),
        ])
        .unwrap();
        let c = AzureAISearchCollection::new(
            AzureAISearchStore::with_api_key("https://s.search.windows.net", "k"),
            "docs".into(),
            definition,
        )
        .unwrap();
        let err = c.build_index().unwrap_err().to_string();
        assert!(
            err.contains("does not support the distance function"),
            "{err}"
        );
    }

    #[test]
    fn a_non_string_key_is_refused() {
        let definition = VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id").with_type("int"),
            VectorStoreField::vector("v", 2),
        ])
        .unwrap();
        let err = AzureAISearchCollection::new(
            AzureAISearchStore::with_api_key("https://s.search.windows.net", "k"),
            "docs".into(),
            definition,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("keys must be Edm.String"), "{err}");
    }

    #[test]
    fn a_field_name_that_is_not_an_identifier_is_refused() {
        let definition = VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id"),
            VectorStoreField::data("content-type"),
        ])
        .unwrap();
        let err = AzureAISearchCollection::new(
            AzureAISearchStore::with_api_key("https://s.search.windows.net", "k"),
            "docs".into(),
            definition,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("simple ASCII identifiers"), "{err}");
    }

    // region: filter translation

    #[test]
    fn scalar_comparisons_translate_to_odata() {
        assert_eq!(
            filter_of(Filter::eq("id", "a").unwrap()).unwrap(),
            "id eq 'a'"
        );
        assert_eq!(
            filter_of(Filter::gte("year", 2020).unwrap()).unwrap(),
            "year ge 2020"
        );
        assert_eq!(
            filter_of(Filter::lt("rating", 4.5).unwrap()).unwrap(),
            "rating lt 4.5"
        );
        assert_eq!(
            filter_of(Filter::between("year", 2020, 2024).unwrap()).unwrap(),
            "(year ge 2020 and year le 2024)"
        );
    }

    #[test]
    fn a_quote_in_a_value_is_doubled_not_injected() {
        // The escaping that keeps a filter value from becoming part of the
        // expression.
        assert_eq!(
            filter_of(Filter::eq("id", "o'brien").unwrap()).unwrap(),
            "id eq 'o''brien'"
        );
        assert_eq!(
            filter_of(Filter::eq("id", "' or id eq '").unwrap()).unwrap(),
            "id eq ''' or id eq '''"
        );
    }

    #[test]
    fn membership_translates_and_not_in_excludes_missing_fields() {
        assert_eq!(
            filter_of(Filter::any_of("year", [json!(2020), json!(2021)]).unwrap()).unwrap(),
            "(year eq 2020 or year eq 2021)"
        );
        assert_eq!(
            filter_of(Filter::none_of("year", [json!(2020)]).unwrap()).unwrap(),
            "(year ne null and not ((year eq 2020)))"
        );
        // An empty `in` matches nothing rather than everything.
        assert_eq!(
            filter_of(Filter::any_of("year", []).unwrap()).unwrap(),
            "false"
        );
    }

    #[test]
    fn collection_membership_uses_any_lambdas() {
        assert_eq!(
            filter_of(Filter::contains("tags", "x").unwrap()).unwrap(),
            "(tags/any(v: v eq 'x'))"
        );
        assert_eq!(
            filter_of(Filter::contains_all("tags", [json!("x"), json!("y")]).unwrap()).unwrap(),
            "(tags/any(v: v eq 'x') and tags/any(v: v eq 'y'))"
        );
        assert_eq!(
            filter_of(Filter::contains_any("tags", [json!("x"), json!("y")]).unwrap()).unwrap(),
            "(tags/any(v: v eq 'x') or tags/any(v: v eq 'y'))"
        );
        let err = filter_of(Filter::contains_all("tags", []).unwrap())
            .unwrap_err()
            .to_string();
        assert!(err.contains("empty 'contains_all'"), "{err}");
        assert_eq!(
            filter_of(Filter::contains_any("tags", []).unwrap()).unwrap(),
            "false"
        );
    }

    #[test]
    fn groups_translate_with_explicit_parentheses() {
        let expr = FilterGroup::and(vec![
            Filter::eq("id", "a").unwrap().into(),
            FilterGroup::not(Filter::eq("year", 2020).unwrap().into()).unwrap(),
        ])
        .unwrap();
        let options = VectorSearchOptions::new(3).with_filter(expr);
        assert_eq!(
            collection().prepare_filter(&options).unwrap().unwrap(),
            "(id eq 'a' and not (year eq 2020))"
        );
    }

    #[test]
    fn the_namespaced_match_operator_becomes_search_ismatch() {
        let filter = Filter::new(
            "text",
            FilterOperator::provider("azure_ai_search.match").unwrap(),
            Some(json!("annual report")),
        )
        .unwrap();
        assert_eq!(
            filter_of(filter).unwrap(),
            "search.ismatch('annual report', 'text', 'simple', 'any')"
        );
    }

    #[test]
    fn match_requires_a_full_text_indexed_field() {
        let filter = Filter::new(
            "year",
            FilterOperator::provider("azure_ai_search.match").unwrap(),
            Some(json!("x")),
        )
        .unwrap();
        let err = filter_of(filter).unwrap_err().to_string();
        assert!(err.contains("full-text-indexed"), "{err}");
    }

    #[test]
    fn operators_azure_cannot_express_are_refused_not_dropped() {
        // Dropping a condition silently widens the result set, which reads as
        // a passing retrieval test while scoping is broken.
        for (filter, expected) in [
            (Filter::exists("year").unwrap(), "missing-versus-null"),
            (Filter::is_null("year").unwrap(), "missing-versus-null"),
            (Filter::is_not_null("year").unwrap(), "missing-versus-null"),
            (Filter::ne("year", 2020).unwrap(), "use a NOT group"),
            (
                Filter::starts_with("text", "a").unwrap(),
                "azure_ai_search.match",
            ),
            (
                Filter::ends_with("text", "a").unwrap(),
                "azure_ai_search.match",
            ),
            (
                Filter::contains_text("text", "a").unwrap(),
                "azure_ai_search.match",
            ),
        ] {
            let err = filter_of(filter).unwrap_err().to_string();
            assert!(err.contains(expected), "expected {expected:?} in {err}");
        }
    }

    #[test]
    fn filtering_on_an_unfilterable_field_is_refused() {
        let err = filter_of(Filter::eq("note", "x").unwrap())
            .unwrap_err()
            .to_string();
        assert!(err.contains("must be filterable"), "{err}");
        let err = filter_of(Filter::eq("embedding", "x").unwrap())
            .unwrap_err()
            .to_string();
        assert!(err.contains("must be filterable"), "{err}");
    }

    #[test]
    fn an_ordered_comparison_on_a_string_field_is_refused() {
        let err = filter_of(Filter::gt("id", "a").unwrap())
            .unwrap_err()
            .to_string();
        assert!(err.contains("only on numeric fields"), "{err}");
    }

    #[test]
    fn an_incompatible_eq_is_false_rather_than_an_error() {
        // Over-specific, not malformed: no record of this type can match.
        assert_eq!(
            filter_of(Filter::eq("year", "not-a-number").unwrap()).unwrap(),
            "false"
        );
        // But an ordered comparison across types is a real mistake.
        let err = filter_of(Filter::gt("year", "x").unwrap())
            .unwrap_err()
            .to_string();
        assert!(err.contains("incompatible"), "{err}");
    }

    #[test]
    fn a_filter_on_an_undeclared_field_is_refused() {
        let err = filter_of(Filter::eq("nope", "x").unwrap())
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("not part of the collection definition"),
            "{err}"
        );
    }

    #[test]
    fn a_provider_filter_passes_through_and_conjoins_with_a_portable_one() {
        let options = VectorSearchOptions::new(3).with_provider_filter("geo.distance(loc, x) lt 1");
        assert_eq!(
            collection().prepare_filter(&options).unwrap().unwrap(),
            "geo.distance(loc, x) lt 1"
        );
        let both = VectorSearchOptions::new(3)
            .with_filter(Filter::eq("id", "a").unwrap())
            .with_provider_filter("geo.distance(loc, x) lt 1");
        assert_eq!(
            collection().prepare_filter(&both).unwrap().unwrap(),
            "(id eq 'a') and (geo.distance(loc, x) lt 1)"
        );
    }

    // region: search body

    #[test]
    fn the_search_body_prefilters_and_widens_k_to_cover_the_window() {
        let options = VectorSearchOptions::new(5)
            .with_skip(10)
            .with_filter(Filter::eq("id", "a").unwrap());
        let body = collection()
            .search_body(&[1.0, 0.0, 0.0], &options, None)
            .unwrap();
        assert_eq!(body["vectorFilterMode"], "preFilter");
        assert_eq!(body["top"], 5);
        assert_eq!(body["skip"], 10);
        // `k` covers skip + top; asking for 5 and then skipping 10 into them
        // would return nothing.
        assert_eq!(body["vectorQueries"][0]["k"], 15);
        assert_eq!(body["vectorQueries"][0]["fields"], "embedding");
        assert_eq!(body["filter"], "id eq 'a'");
        // Vectors are not projected unless asked for.
        assert_eq!(body["select"], "id,text,year,rating,tags,note");
    }

    #[test]
    fn include_vectors_projects_the_vector_field() {
        let options = VectorSearchOptions::new(1).with_include_vectors(true);
        let body = collection()
            .search_body(&[1.0, 0.0, 0.0], &options, None)
            .unwrap();
        assert_eq!(body["select"], "id,text,year,rating,tags,note,embedding");
    }

    #[test]
    fn a_query_vector_of_the_wrong_width_is_rejected_before_the_request() {
        let err = collection()
            .search_body(&[1.0, 0.0], &VectorSearchOptions::new(1), None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("declares 3"), "{err}");
    }

    #[test]
    fn a_non_finite_query_vector_is_rejected() {
        let err = collection()
            .search_body(&[1.0, f32::NAN, 0.0], &VectorSearchOptions::new(1), None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("not finite"), "{err}");
    }

    #[test]
    fn the_result_window_ceiling_is_enforced() {
        let options = VectorSearchOptions::new(10).with_skip(10_000);
        let err = collection()
            .search_body(&[1.0, 0.0, 0.0], &options, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("result window"), "{err}");
    }

    #[test]
    fn a_hybrid_search_names_the_full_text_fields() {
        let body = collection()
            .search_body(
                &[1.0, 0.0, 0.0],
                &VectorSearchOptions::new(3),
                Some("report"),
            )
            .unwrap();
        assert_eq!(body["search"], "report");
        assert_eq!(body["searchFields"], "text");
    }

    #[test]
    fn a_hybrid_search_without_a_full_text_field_is_refused() {
        let definition = VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id"),
            VectorStoreField::vector("v", 2),
        ])
        .unwrap();
        let c = AzureAISearchCollection::new(
            AzureAISearchStore::with_api_key("https://s.search.windows.net", "k"),
            "docs".into(),
            definition,
        )
        .unwrap();
        let err = c
            .search_body(&[1.0, 0.0], &VectorSearchOptions::new(3), Some("x"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("full-text-indexed"), "{err}");
    }

    #[test]
    fn urls_carry_the_api_version_exactly_once() {
        let store = AzureAISearchStore::with_api_key("https://s.search.windows.net/", "k");
        assert_eq!(
            store.url("indexes('docs')"),
            "https://s.search.windows.net/indexes('docs')?api-version=2024-07-01"
        );
        assert_eq!(
            store.url("indexes?$select=name"),
            "https://s.search.windows.net/indexes?$select=name&api-version=2024-07-01"
        );
    }
}
