//! A caller-owned vector collection, exposed to an agent as tools.
//!
//! [`VectorCollectionContextProvider`] turns any [`VectorCollection`] — the
//! in-memory one, Azure AI Search, Azure Cosmos DB — into a set of tools the
//! model can call: search by similarity, read by key, write, delete. The
//! collection, its schema and its record lifecycle stay with the caller; this
//! only puts a door on them.
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use agent_framework_core::vectors::{
//! #     InMemoryVectorStore, VectorCollectionContextProvider, VectorStore,
//! #     VectorStoreCollectionDefinition, VectorStoreField, Filter,
//! # };
//! # use agent_framework_core::client::EmbeddingClient;
//! # async fn demo(embedder: Arc<dyn EmbeddingClient>) -> agent_framework_core::error::Result<()> {
//! let definition = VectorStoreCollectionDefinition::new(vec![
//!     VectorStoreField::key("id").with_type("str"),
//!     VectorStoreField::data("text").with_type("str"),
//!     VectorStoreField::data("tenant").with_type("str"),
//!     VectorStoreField::vector("embedding", 1536),
//! ])?;
//! let store = InMemoryVectorStore::new();
//! let collection = store.get_collection("notes", definition)?;
//!
//! let provider = VectorCollectionContextProvider::builder(collection.into(), embedder)
//!     // Best-effort grouping, not an authorization boundary — see below.
//!     .scope_filter(Filter::eq("tenant", "acme")?)
//!     .embed_from_field("text")
//!     .build()?;
//! # let _ = provider;
//! # Ok(())
//! # }
//! ```
//!
//! # The scope filter is grouping, not authorization
//!
//! `scope_filter` is conjoined into every generated tool's search, and checked
//! locally on the records a read, write or delete touches. That makes it a
//! reliable way to keep one agent's records apart from another's *when both
//! are cooperating*. It is not a security boundary: the check on a delete is
//! read-then-check-then-delete, which is not atomic, and a caller-supplied
//! additional tool carries whatever filter the caller gave it. Upstream
//! documents the same limits. Where scoping has to hold against a hostile
//! party, give each party its own collection.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Map, Value};

use super::{
    FieldType, FilterExpression, VectorCollection, VectorSearchOptions,
    VectorStoreCollectionDefinition,
};
use crate::client::EmbeddingClient;
use crate::error::{Error, Result};
use crate::memory::{ContextProvider, SessionContext};
use crate::tools::{ApprovalMode, FunctionTool, ToolDefinition};

/// The default cap on how many records or keys one tool call may carry.
///
/// A model asked to "delete the old ones" will happily name two hundred keys;
/// each is a round trip to the store, and a batch that large is more often a
/// misunderstanding than an intent.
pub const DEFAULT_MAX_TOOL_BATCH_SIZE: usize = 10;

/// The default number of hits the search tool returns.
pub const DEFAULT_SEARCH_TOP: usize = 5;

/// Which generated tool an approval mode applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorToolKind {
    Search,
    Get,
    Upsert,
    Delete,
}

impl VectorToolKind {
    fn tool_name(self) -> &'static str {
        match self {
            Self::Search => "search",
            Self::Get => "get",
            Self::Upsert => "upsert",
            Self::Delete => "delete",
        }
    }

    /// The default approval mode, matching upstream's: reads are free, writes
    /// are not. A model that decides to tidy up a collection should have to
    /// ask first, and a delete it gets wrong is not recoverable from the
    /// conversation.
    pub fn default_approval(self) -> ApprovalMode {
        match self {
            Self::Search | Self::Get => ApprovalMode::NeverRequire,
            Self::Upsert | Self::Delete => ApprovalMode::AlwaysRequire,
        }
    }
}

/// Builds a [`VectorCollectionContextProvider`].
pub struct VectorCollectionContextProviderBuilder {
    collection: Arc<dyn VectorCollection>,
    embedder: Arc<dyn EmbeddingClient>,
    source_id: String,
    tool_prefix: Option<String>,
    scope_filter: Option<FilterExpression>,
    instructions: Option<Vec<String>>,
    include_search: bool,
    include_get: bool,
    include_delete: bool,
    embed_from_field: Option<String>,
    vector_field: Option<String>,
    approvals: Vec<(VectorToolKind, ApprovalMode)>,
    max_tool_batch_size: usize,
    search_top: usize,
    additional_tools: Vec<ToolDefinition>,
}

impl VectorCollectionContextProviderBuilder {
    /// Identify this provider in instruction and tool attribution. Defaults to
    /// `"vector_collection"`.
    pub fn source_id(mut self, source_id: impl Into<String>) -> Self {
        self.source_id = source_id.into();
        self
    }

    /// Prefix every generated tool's name, e.g. `"notes"` gives
    /// `notes_search`.
    ///
    /// The generated names are otherwise the plain `search` / `get` /
    /// `upsert` / `delete` upstream uses, which collide when an agent holds
    /// two of these providers — or one of these and any other tool called
    /// `search`. A collision is not a silent problem (the invocation loop
    /// resolves a call by name and would pick one of them), which is exactly
    /// why it is worth avoiding.
    pub fn tool_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.tool_prefix = Some(prefix.into());
        self
    }

    /// Scope every generated tool to the records matching `filter`. See the
    /// module docs for what this does and does not guarantee.
    pub fn scope_filter(mut self, filter: impl Into<FilterExpression>) -> Self {
        self.scope_filter = Some(filter.into());
        self
    }

    /// Replace the generated instructions. An empty list adds none.
    pub fn instructions(mut self, instructions: Vec<String>) -> Self {
        self.instructions = Some(instructions);
        self
    }

    /// Enable the `upsert` tool, embedding the named **data** field's text
    /// into the collection's vector field.
    ///
    /// Without this there is no upsert tool, and `build` says so rather than
    /// offering one that cannot work: a model cannot author a 1,536-float
    /// embedding, and which of a collection's fields carries the text it
    /// should be derived from is a property of the schema that nothing in the
    /// definition records. Upstream has no equivalent because its collection
    /// owns an embedding generator and does this itself.
    pub fn embed_from_field(mut self, field_name: impl Into<String>) -> Self {
        self.embed_from_field = Some(field_name.into());
        self
    }

    /// Name the vector field the generated tools use.
    ///
    /// Only needed when the collection declares more than one, where there is
    /// no safe default: searching the wrong embedding returns confident
    /// nonsense rather than an error, so [`build`](Self::build) refuses a
    /// multi-vector collection that does not name one.
    pub fn vector_field(mut self, name: impl Into<String>) -> Self {
        self.vector_field = Some(name.into());
        self
    }

    /// Whether to generate the similarity-search tool. On by default.
    pub fn include_search(mut self, include: bool) -> Self {
        self.include_search = include;
        self
    }

    /// Whether to generate the read-by-key tool. On by default.
    pub fn include_get(mut self, include: bool) -> Self {
        self.include_get = include;
        self
    }

    /// Whether to generate the delete-by-key tool. On by default.
    pub fn include_delete(mut self, include: bool) -> Self {
        self.include_delete = include;
        self
    }

    /// Override one tool's approval mode. See
    /// [`VectorToolKind::default_approval`] for the defaults.
    pub fn approval_mode(mut self, kind: VectorToolKind, mode: ApprovalMode) -> Self {
        self.approvals.push((kind, mode));
        self
    }

    /// Cap how many records or keys one tool call may carry. See
    /// [`DEFAULT_MAX_TOOL_BATCH_SIZE`].
    pub fn max_tool_batch_size(mut self, max: usize) -> Self {
        self.max_tool_batch_size = max;
        self
    }

    /// How many hits the search tool returns. See [`DEFAULT_SEARCH_TOP`].
    pub fn search_top(mut self, top: usize) -> Self {
        self.search_top = top;
        self
    }

    /// Add a caller-built tool alongside the generated ones.
    ///
    /// These are passed through untouched — in particular the scope filter is
    /// **not** applied to them, so a tool that searches a shared collection
    /// has to carry its own.
    pub fn additional_tool(mut self, tool: ToolDefinition) -> Self {
        self.additional_tools.push(tool);
        self
    }

    /// Validate the configuration and build the provider.
    pub fn build(self) -> Result<VectorCollectionContextProvider> {
        if self.max_tool_batch_size == 0 {
            return Err(Error::Configuration(
                "max_tool_batch_size must be greater than zero".into(),
            ));
        }
        if self.search_top == 0 {
            return Err(Error::Configuration(
                "search_top must be greater than zero".into(),
            ));
        }
        let definition = self.collection.definition().clone();

        let key_type = definition.key_field().type_.clone();

        let mut embed_source = None;
        if let Some(name) = &self.embed_from_field {
            let field = definition.try_get_field(name).ok_or_else(|| {
                Error::Configuration(format!(
                    "embed_from_field names '{name}', which the collection does not declare"
                ))
            })?;
            if field.field_type != FieldType::Data {
                return Err(Error::Configuration(format!(
                    "embed_from_field must name a data field; '{name}' is the collection's {:?} \
                     field",
                    field.field_type
                )));
            }
            // A declared non-string type cannot work: the generated schema
            // would ask the model for that type, and the executor reads the
            // field with `as_str`, so every schema-valid upsert would fail at
            // runtime. Refused here instead, where the caller can see it. An
            // *undeclared* type is left alone — the executor's own check
            // covers it, and guessing would reject a perfectly good untyped
            // text field.
            if let Some(declared) = field.type_.as_deref() {
                if !matches!(
                    declared.trim().to_ascii_lowercase().as_str(),
                    "str" | "string"
                ) {
                    return Err(Error::Configuration(format!(
                        "embed_from_field names '{name}', which the collection declares as \
                         '{declared}'; an embedding is derived from text, so this field has to \
                         be a string"
                    )));
                }
            }
            embed_source = Some(name.clone());
        }

        // A filter the enabled tools cannot honor would fail — or silently
        // match nothing — at the first tool call rather than here, where the
        // caller can see it.
        if let Some(filter) = &self.scope_filter {
            filter.validate()?;
            let locally_evaluated =
                self.include_get || self.include_delete || embed_source.is_some();
            validate_filter_fields(filter, &definition, locally_evaluated)?;
        }

        // Only search and upsert touch a vector; get and delete work by key.
        // Resolving unconditionally would refuse a read/delete-only provider
        // over a collection with no vector field, or with several and none
        // named — configurations the underlying `VectorCollection` serves
        // perfectly well. Resolved here rather than at each tool call,
        // because picking the first of several silently would search or write
        // whichever happened to be declared first, and leaving it unset would
        // fail every search on a multi-vector collection.
        let vector_field = if self.include_search || embed_source.is_some() {
            Some(
                definition
                    .try_get_vector_field(self.vector_field.as_deref())
                    .cloned()
                    .ok_or_else(|| {
                        Error::Configuration(match &self.vector_field {
                            Some(name) => format!(
                                "vector_field names '{name}', which is not one of this \
                                 collection's vector fields"
                            ),
                            None if definition.vector_fields().is_empty() => {
                                "a search or upsert tool needs a collection with a vector field; \
                                 turn them off to build a read/delete-only provider"
                                    .into()
                            }
                            None => format!(
                                "this collection declares {} vector fields, so the tools have to \
                                 be told which one to use: set `vector_field`",
                                definition.vector_fields().len()
                            ),
                        })
                    })?,
            )
        } else {
            None
        };

        let approval = |kind: VectorToolKind| {
            self.approvals
                .iter()
                .rev()
                .find(|(k, _)| *k == kind)
                .map(|(_, m)| *m)
                .unwrap_or_else(|| kind.default_approval())
        };
        let name_of = |kind: VectorToolKind| match &self.tool_prefix {
            Some(prefix) => format!("{prefix}_{}", kind.tool_name()),
            None => kind.tool_name().to_string(),
        };

        let mut tools: Vec<ToolDefinition> = Vec::new();
        if self.include_search {
            tools.push(build_search_tool(
                Arc::clone(&self.collection),
                Arc::clone(&self.embedder),
                self.scope_filter.clone(),
                self.search_top,
                vector_field
                    .as_ref()
                    .expect("search implies a resolved vector field")
                    .name
                    .clone(),
                name_of(VectorToolKind::Search),
                approval(VectorToolKind::Search),
            ));
        }
        if self.include_get {
            tools.push(build_get_tool(
                Arc::clone(&self.collection),
                self.scope_filter.clone(),
                self.max_tool_batch_size,
                key_type.clone(),
                name_of(VectorToolKind::Get),
                approval(VectorToolKind::Get),
            ));
        }
        if let Some(text_field) = embed_source {
            tools.push(build_upsert_tool(
                Arc::clone(&self.collection),
                Arc::clone(&self.embedder),
                self.scope_filter.clone(),
                self.max_tool_batch_size,
                text_field,
                vector_field
                    .as_ref()
                    .expect("an embedding source implies a resolved vector field")
                    .name
                    .clone(),
                name_of(VectorToolKind::Upsert),
                approval(VectorToolKind::Upsert),
            )?);
        }
        if self.include_delete {
            tools.push(build_delete_tool(
                Arc::clone(&self.collection),
                self.scope_filter.clone(),
                self.max_tool_batch_size,
                key_type.clone(),
                name_of(VectorToolKind::Delete),
                approval(VectorToolKind::Delete),
            ));
        }
        tools.extend(self.additional_tools);

        let mut seen = std::collections::HashSet::new();
        for tool in &tools {
            if !seen.insert(tool.name.clone()) {
                return Err(Error::Configuration(format!(
                    "two of this provider's tools are named '{}'; set a `tool_prefix` or rename \
                     the additional tool",
                    tool.name
                )));
            }
        }

        let instructions = self
            .instructions
            .unwrap_or_else(|| default_instructions(&tools));

        Ok(VectorCollectionContextProvider {
            source_id: self.source_id,
            instructions,
            tools,
        })
    }
}

/// Exposes a vector collection to an agent as tools. See the module docs.
pub struct VectorCollectionContextProvider {
    source_id: String,
    instructions: Vec<String>,
    tools: Vec<ToolDefinition>,
}

impl std::fmt::Debug for VectorCollectionContextProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorCollectionContextProvider")
            .field("source_id", &self.source_id)
            .field(
                "tools",
                &self.tools.iter().map(|t| &t.name).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl VectorCollectionContextProvider {
    /// The default [`source_id`](VectorCollectionContextProviderBuilder::source_id).
    pub const DEFAULT_SOURCE_ID: &'static str = "vector_collection";

    /// Start building a provider over `collection`.
    ///
    /// `embedder` turns a model's query text — and, with
    /// [`embed_from_field`](VectorCollectionContextProviderBuilder::embed_from_field),
    /// a record's text — into a vector. It is required because this port's
    /// [`VectorCollection::search`] takes a vector, where upstream's
    /// collection owns an embedding generator and embeds internally.
    pub fn builder(
        collection: Arc<dyn VectorCollection>,
        embedder: Arc<dyn EmbeddingClient>,
    ) -> VectorCollectionContextProviderBuilder {
        VectorCollectionContextProviderBuilder {
            collection,
            embedder,
            source_id: Self::DEFAULT_SOURCE_ID.to_string(),
            tool_prefix: None,
            scope_filter: None,
            instructions: None,
            include_search: true,
            include_get: true,
            include_delete: true,
            embed_from_field: None,
            vector_field: None,
            approvals: Vec::new(),
            max_tool_batch_size: DEFAULT_MAX_TOOL_BATCH_SIZE,
            search_top: DEFAULT_SEARCH_TOP,
            additional_tools: Vec::new(),
        }
    }

    /// The tools this provider contributes to a run.
    pub fn tools(&self) -> &[ToolDefinition] {
        &self.tools
    }

    /// The instructions this provider contributes to a run.
    pub fn instructions(&self) -> &[String] {
        &self.instructions
    }
}

#[async_trait]
impl ContextProvider for VectorCollectionContextProvider {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        for instruction in &self.instructions {
            ctx.add_instructions(instruction.clone());
        }
        ctx.tools.extend(self.tools.iter().cloned());
        Ok(())
    }
}

// region: tool construction

fn default_instructions(tools: &[ToolDefinition]) -> Vec<String> {
    let has = |suffix: &str| {
        tools
            .iter()
            .any(|t| t.name == suffix || t.name.ends_with(&format!("_{suffix}")))
    };
    let mut out = vec![
        "Use the available vector collection tools as the source of truth for stored records."
            .to_string(),
    ];
    if has("get") && has("search") {
        out.push(
            "Use get when record keys are known; use search to find records by similarity."
                .to_string(),
        );
    }
    if has("upsert") {
        out.push("Use upsert to create or update collection records.".to_string());
    }
    if has("delete") {
        out.push("Use delete only when collection records should be removed.".to_string());
    }
    out
}

/// Refuse a scope filter the enabled tools cannot actually honor.
///
/// Every leaf must name a declared field. Beyond that, `locally_evaluated`
/// asks for more, because `get`, `delete` and `upsert` check the scope by
/// running [`FilterExpression::matches`] over a record in this process rather
/// than handing the filter to the store:
///
/// * A **provider operator** (`azure_ai_search.match`) has no local meaning —
///   `matches` errors on it — so those tools would fail on every call while
///   search, which passes the filter to the connector, worked fine.
/// * A **vector field** predicate cannot be answered either: `get` and
///   `delete` fetch with `include_vectors = false`, and `upsert` checks the
///   scope before the embedding it is about to derive exists. A missing field
///   is a non-match, so instead of failing, those tools would quietly report
///   every record as out of scope — hiding reads and refusing writes with no
///   error to explain it.
///
/// Refused here rather than discovered there. A search-only provider may
/// still carry either, since nothing evaluates it locally.
fn validate_filter_fields(
    filter: &FilterExpression,
    definition: &VectorStoreCollectionDefinition,
    locally_evaluated: bool,
) -> Result<()> {
    match filter {
        FilterExpression::Group(group) => {
            for child in &group.filters {
                validate_filter_fields(child, definition, locally_evaluated)?;
            }
            Ok(())
        }
        FilterExpression::Condition(condition) => {
            let Some(field) = definition.try_get_field(&condition.field_name) else {
                return Err(Error::Configuration(format!(
                    "scope filter names field '{}', which the collection does not declare",
                    condition.field_name
                )));
            };
            if !locally_evaluated {
                return Ok(());
            }
            if !condition.operator.is_standard() {
                return Err(Error::Configuration(format!(
                    "scope filter uses the provider-specific operator '{}', which the get, \
                     delete and upsert tools cannot evaluate; use a portable operator, or turn \
                     those tools off and keep only search",
                    condition.operator.as_str()
                )));
            }
            if field.field_type == FieldType::Vector {
                return Err(Error::Configuration(format!(
                    "scope filter tests the vector field '{}', which the get, delete and upsert \
                     tools cannot evaluate — they never hold the vector — so every record would \
                     read as out of scope; scope on a data field instead",
                    condition.field_name
                )));
            }
            Ok(())
        }
    }
}

/// Conjoin the scope filter into a search's options.
fn scoped_options(
    top: usize,
    scope_filter: &Option<FilterExpression>,
    vector_field: &str,
) -> Result<VectorSearchOptions> {
    // Always named, even on a single-vector collection: it costs nothing
    // there and it is the difference between working and failing on a
    // collection with several.
    let mut options = VectorSearchOptions::new(top).with_vector_field_name(vector_field);
    if let Some(filter) = scope_filter {
        options = options.with_filter(filter.clone());
    }
    Ok(options)
}

/// Whether `record` (keyed by logical name) is inside the scope.
///
/// `matches` reads storage names, and the records these tools hold are keyed
/// by logical name, so the resolver is the identity rather than the
/// definition's renaming.
fn in_scope(record: &Value, scope_filter: &Option<FilterExpression>) -> Result<bool> {
    match scope_filter {
        None => Ok(true),
        Some(filter) => filter.matches(record, &|name: &str| Some(name.to_string())),
    }
}

/// The `{ "keys": [...] }` schema for the read and delete tools.
///
/// The item type comes from the *key field's* declared type rather than
/// being hardcoded to `string`: a collection's key is a `serde_json::Value`,
/// and a key declared `int` is stored and looked up as the JSON number `42`,
/// not the string `"42"` — `InMemoryVectorStore` keys its map on
/// `Value::to_string()`, so the two do not collide, they simply never match.
/// Telling the model `string` on such a collection guarantees every read and
/// delete silently finds nothing. An undeclared key type stays unconstrained.
fn keys_schema(description: &str, max_items: usize, key_type: Option<&str>) -> Value {
    json!({
        "type": "object",
        "properties": {
            "keys": {
                "type": "array",
                "description": description,
                "items": { "type": json_type_for(key_type) },
                "maxItems": max_items,
            }
        },
        "required": ["keys"],
    })
}

/// Drop repeated keys, preserving order.
///
/// A model that names the same record twice would otherwise have it counted
/// twice in the delete tally, reporting two deletions for one record.
/// Compared by JSON rendering, which is how `InMemoryVectorStore` keys its
/// own map — `"42"` and `42` are different keys there and stay different
/// here.
fn dedup_keys(keys: Vec<Value>) -> Vec<Value> {
    let mut seen = std::collections::HashSet::new();
    keys.into_iter()
        .filter(|key| seen.insert(key.to_string()))
        .collect()
}

/// Read a `keys` argument, enforcing the batch cap.
fn keys_argument(args: &Value, max_batch_size: usize) -> Result<Vec<Value>> {
    let keys = args
        .get("keys")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Tool("`keys` must be an array of record keys".into()))?;
    if keys.is_empty() {
        return Err(Error::Tool("`keys` must name at least one record".into()));
    }
    if keys.len() > max_batch_size {
        return Err(Error::Tool(format!(
            "`keys` carries {} entries, above this tool's limit of {max_batch_size}",
            keys.len()
        )));
    }
    Ok(keys.clone())
}

#[allow(clippy::too_many_arguments)]
fn build_search_tool(
    collection: Arc<dyn VectorCollection>,
    embedder: Arc<dyn EmbeddingClient>,
    scope_filter: Option<FilterExpression>,
    top: usize,
    vector_field: String,
    name: String,
    approval: ApprovalMode,
) -> ToolDefinition {
    let schema = json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to look for. Matched by meaning, not by keyword.",
            }
        },
        "required": ["query"],
    });
    FunctionTool::new(
        name,
        "Find the stored records most similar in meaning to a query.",
        schema,
        move |args: Value| {
            let collection = Arc::clone(&collection);
            let embedder = Arc::clone(&embedder);
            let scope_filter = scope_filter.clone();
            let vector_field = vector_field.clone();
            async move {
                let query = args
                    .get("query")
                    .and_then(Value::as_str)
                    .filter(|q| !q.trim().is_empty())
                    .ok_or_else(|| Error::Tool("`query` must be a non-empty string".into()))?;
                let embeddings = embedder
                    .get_embeddings(vec![query.to_string()], None)
                    .await?;
                let vector = embeddings
                    .embeddings
                    .into_iter()
                    .next()
                    .ok_or_else(|| {
                        Error::Tool("the embedding service returned no vector for the query".into())
                    })?
                    .vector;
                let options = scoped_options(top, &scope_filter, &vector_field)?;
                let hits = collection.search(vector, &options).await?;
                Ok(json!({
                    "results": hits
                        .into_iter()
                        .map(|hit| json!({ "record": hit.record, "score": hit.score }))
                        .collect::<Vec<_>>(),
                }))
            }
        },
    )
    .with_approval_mode(approval)
    .into_definition()
}

fn build_get_tool(
    collection: Arc<dyn VectorCollection>,
    scope_filter: Option<FilterExpression>,
    max_batch_size: usize,
    key_type: Option<String>,
    name: String,
    approval: ApprovalMode,
) -> ToolDefinition {
    let schema = keys_schema(
        "The keys of the records to read.",
        max_batch_size,
        key_type.as_deref(),
    );
    FunctionTool::new(
        name,
        "Read stored records by key.",
        schema,
        move |args: Value| {
            let collection = Arc::clone(&collection);
            let scope_filter = scope_filter.clone();
            async move {
                let keys = keys_argument(&args, max_batch_size)?;
                let found = collection.get(keys.clone(), false).await?;
                let mut records = Vec::with_capacity(found.len());
                for (key, record) in keys.iter().zip(found) {
                    // A record outside the scope reads as absent rather than
                    // as a refusal: telling the model that a key it may not
                    // read exists is itself a disclosure.
                    let visible = match &record {
                        Some(r) => in_scope(r, &scope_filter)?,
                        None => false,
                    };
                    records.push(json!({
                        "key": key,
                        "record": if visible { record } else { None },
                    }));
                }
                Ok(json!({ "records": records }))
            }
        },
    )
    .with_approval_mode(approval)
    .into_definition()
}

#[allow(clippy::too_many_arguments)]
fn build_upsert_tool(
    collection: Arc<dyn VectorCollection>,
    embedder: Arc<dyn EmbeddingClient>,
    scope_filter: Option<FilterExpression>,
    max_batch_size: usize,
    text_field: String,
    vector_field: String,
    name: String,
    approval: ApprovalMode,
) -> Result<ToolDefinition> {
    let definition = collection.definition().clone();
    let key_field = definition.key_field().name.clone();
    let mut properties = Map::new();
    let mut required = Vec::new();
    for field in definition.fields() {
        // The vector is derived from `text_field`, never authored: a model
        // cannot produce an embedding, and one it invented would poison the
        // index rather than fail loudly.
        if field.field_type == FieldType::Vector {
            continue;
        }
        properties.insert(
            field.name.clone(),
            json!({ "type": json_type_for(field.type_.as_deref()) }),
        );
        if field.field_type == FieldType::Key {
            required.push(Value::String(field.name.clone()));
        }
    }
    required.push(Value::String(text_field.clone()));
    let schema = json!({
        "type": "object",
        "properties": {
            "records": {
                "type": "array",
                "description": "The records to create or replace. Each replaces any record with the same key in full.",
                "maxItems": max_batch_size,
                "items": {
                    "type": "object",
                    "properties": Value::Object(properties),
                    "required": Value::Array(required),
                },
            }
        },
        "required": ["records"],
    });

    Ok(FunctionTool::new(
        name,
        "Create or replace stored records.",
        schema,
        move |args: Value| {
            let collection = Arc::clone(&collection);
            let embedder = Arc::clone(&embedder);
            let scope_filter = scope_filter.clone();
            let text_field = text_field.clone();
            let vector_field = vector_field.clone();
            let key_field = key_field.clone();
            async move {
                let records = args
                    .get("records")
                    .and_then(Value::as_array)
                    .ok_or_else(|| Error::Tool("`records` must be an array".into()))?;
                if records.is_empty() {
                    return Err(Error::Tool(
                        "`records` must carry at least one record".into(),
                    ));
                }
                if records.len() > max_batch_size {
                    return Err(Error::Tool(format!(
                        "`records` carries {} entries, above this tool's limit of \
                         {max_batch_size}",
                        records.len()
                    )));
                }

                let mut texts = Vec::with_capacity(records.len());
                for (index, record) in records.iter().enumerate() {
                    if !in_scope(record, &scope_filter)? {
                        // Refused rather than rewritten: silently stamping the
                        // scope onto the record would let the model write
                        // whatever it liked and have the provider launder it.
                        return Err(Error::Tool(format!(
                            "the record at index {index} falls outside this tool's scope"
                        )));
                    }
                    let text = record
                        .get(&text_field)
                        .and_then(Value::as_str)
                        .filter(|t| !t.trim().is_empty())
                        .ok_or_else(|| {
                            Error::Tool(format!(
                                "the record at index {index} has no text in '{text_field}', which \
                                 is the field its embedding is derived from"
                            ))
                        })?;
                    texts.push(text.to_string());
                }

                // An upsert *replaces* the whole document at its key, so a
                // payload that is in scope says nothing about the record it
                // lands on. Without this, a scoped agent overwrites another
                // group's record by naming its key — laundering by key
                // collision rather than by payload, and destroying a record
                // `get` and `delete` refuse to show it. Read, check, then
                // write; not atomic, for the reason the module docs give.
                if scope_filter.is_some() {
                    let keys: Vec<Value> = records
                        .iter()
                        .map(|record| record.get(&key_field).cloned().unwrap_or(Value::Null))
                        .collect();
                    let existing = collection.get(keys, false).await?;
                    for (index, current) in existing.into_iter().enumerate() {
                        let Some(current) = current else {
                            continue;
                        };
                        if !in_scope(&current, &scope_filter)? {
                            return Err(Error::Tool(format!(
                                "the record at index {index} would replace an existing record \
                                 outside this tool's scope"
                            )));
                        }
                    }
                }

                let embeddings = embedder.get_embeddings(texts, None).await?;
                if embeddings.embeddings.len() != records.len() {
                    return Err(Error::Tool(format!(
                        "the embedding service returned {} vectors for {} records",
                        embeddings.embeddings.len(),
                        records.len()
                    )));
                }

                let prepared: Vec<Value> = records
                    .iter()
                    .zip(embeddings.embeddings)
                    .map(|(record, embedding)| {
                        let mut object = record.as_object().cloned().unwrap_or_default();
                        object.insert(vector_field.clone(), json!(embedding.vector));
                        Value::Object(object)
                    })
                    .collect();
                let keys = collection.upsert(prepared).await?;
                Ok(json!({ "keys": keys }))
            }
        },
    )
    .with_approval_mode(approval)
    .into_definition())
}

fn build_delete_tool(
    collection: Arc<dyn VectorCollection>,
    scope_filter: Option<FilterExpression>,
    max_batch_size: usize,
    key_type: Option<String>,
    name: String,
    approval: ApprovalMode,
) -> ToolDefinition {
    let schema = keys_schema(
        "The keys of the records to delete.",
        max_batch_size,
        key_type.as_deref(),
    );
    FunctionTool::new(
        name,
        "Delete stored records by key.",
        schema,
        move |args: Value| {
            let collection = Arc::clone(&collection);
            let scope_filter = scope_filter.clone();
            async move {
                let keys = dedup_keys(keys_argument(&args, max_batch_size)?);
                // Read, check, then delete — on both paths. Not atomic, a
                // record can move out of scope or be removed in between,
                // which is why the module docs call this grouping rather than
                // authorization. The read is not only for the scope check:
                // without it an absent key counts as deleted, which is the
                // opposite of what the counts below promise.
                let found = collection.get(keys.clone(), false).await?;
                let mut deletable = Vec::new();
                for (key, record) in keys.iter().zip(found) {
                    if let Some(record) = record {
                        if in_scope(&record, &scope_filter)? {
                            deletable.push(key.clone());
                        }
                    }
                }
                let deleted = deletable.len();
                if !deletable.is_empty() {
                    collection.delete(deletable).await?;
                }
                Ok(json!({
                    "deleted": deleted,
                    // A key that was already gone and one outside the scope
                    // are both "not deleted"; the model does not need to be
                    // told which, and telling it would disclose the second.
                    "not_deleted": keys.len() - deleted,
                }))
            }
        },
    )
    .with_approval_mode(approval)
    .into_definition()
}

/// The JSON Schema type for a field's free-form type hint. Anything
/// unrecognized is left unconstrained rather than guessed at.
fn json_type_for(hint: Option<&str>) -> Value {
    match hint.map(|h| h.trim().to_ascii_lowercase()).as_deref() {
        Some("str" | "string") => json!("string"),
        Some("int" | "integer" | "int32" | "int64") => json!("integer"),
        Some("float" | "double" | "number") => json!("number"),
        Some("bool" | "boolean") => json!("boolean"),
        Some("list" | "tuple" | "set" | "sequence" | "array") => json!("array"),
        Some("dict" | "object" | "map") => json!("object"),
        _ => json!(["string", "number", "integer", "boolean", "array", "object", "null"]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Embedding, EmbeddingGenerationOptions, GeneratedEmbeddings};
    use crate::vectors::{Filter, InMemoryVectorStore, VectorStore, VectorStoreField};

    /// A deterministic stand-in: each value becomes a three-dimensional vector
    /// from its length, so similarity is a property of the test's inputs
    /// rather than of a service.
    struct StubEmbedder;

    #[async_trait]
    impl EmbeddingClient for StubEmbedder {
        async fn get_embeddings(
            &self,
            values: Vec<String>,
            _options: Option<EmbeddingGenerationOptions>,
        ) -> Result<GeneratedEmbeddings> {
            Ok(GeneratedEmbeddings {
                embeddings: values
                    .iter()
                    .map(|v| Embedding {
                        vector: vec![v.len() as f32, 1.0, 0.0],
                        ..Default::default()
                    })
                    .collect(),
                ..Default::default()
            })
        }
    }

    fn definition() -> VectorStoreCollectionDefinition {
        VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id").with_type("str"),
            VectorStoreField::data("text").with_type("str"),
            VectorStoreField::data("tenant").with_type("str"),
            VectorStoreField::vector("embedding", 3),
        ])
        .unwrap()
    }

    fn collection() -> Arc<dyn VectorCollection> {
        let store = InMemoryVectorStore::new();
        Arc::from(store.get_collection("notes", definition()).unwrap())
    }

    fn builder(collection: Arc<dyn VectorCollection>) -> VectorCollectionContextProviderBuilder {
        VectorCollectionContextProvider::builder(collection, Arc::new(StubEmbedder))
    }

    async fn call(provider: &VectorCollectionContextProvider, name: &str, args: Value) -> Value {
        let tool = provider
            .tools()
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("no tool named {name}"));
        let executor = tool.executor.as_ref().expect("a generated tool executes");
        executor.invoke(args).await.expect("the tool call succeeds")
    }

    #[test]
    fn writes_require_approval_by_default_and_reads_do_not() {
        // A delete the model gets wrong is not recoverable from the
        // conversation, so it asks first.
        let provider = builder(collection())
            .embed_from_field("text")
            .build()
            .unwrap();
        for (name, expected) in [
            ("search", ApprovalMode::NeverRequire),
            ("get", ApprovalMode::NeverRequire),
            ("upsert", ApprovalMode::AlwaysRequire),
            ("delete", ApprovalMode::AlwaysRequire),
        ] {
            let tool = provider.tools().iter().find(|t| t.name == name).unwrap();
            assert_eq!(tool.approval_mode, expected, "{name}");
        }
    }

    #[test]
    fn without_a_text_field_there_is_no_upsert_tool() {
        // Rather than one the model cannot use: it cannot author an
        // embedding, and nothing in the definition says which field to derive
        // one from.
        let provider = builder(collection()).build().unwrap();
        assert!(provider.tools().iter().all(|t| t.name != "upsert"));
    }

    #[test]
    fn a_tool_prefix_keeps_two_providers_apart() {
        let provider = builder(collection()).tool_prefix("notes").build().unwrap();
        let names: Vec<_> = provider.tools().iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["notes_search", "notes_get", "notes_delete"]);
    }

    #[test]
    fn a_duplicate_tool_name_is_refused_at_build() {
        let duplicate = FunctionTool::new(
            "search",
            "another search",
            json!({ "type": "object", "properties": {} }),
            |_args| async move { Ok(json!({})) },
        )
        .into_definition();
        let err = builder(collection())
            .additional_tool(duplicate)
            .build()
            .unwrap_err()
            .to_string();
        assert!(err.contains("named 'search'"), "{err}");
    }

    fn two_vector_definition() -> VectorStoreCollectionDefinition {
        VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id").with_type("str"),
            VectorStoreField::data("text").with_type("str"),
            VectorStoreField::vector("title_embedding", 3),
            VectorStoreField::vector("body_embedding", 3),
        ])
        .unwrap()
    }

    fn two_vector_collection() -> Arc<dyn VectorCollection> {
        let store = InMemoryVectorStore::new();
        Arc::from(
            store
                .get_collection("notes", two_vector_definition())
                .unwrap(),
        )
    }

    #[test]
    fn a_multi_vector_collection_must_say_which_field_the_tools_use() {
        // Picking the first silently would search or write whichever field
        // happened to be declared first; leaving it unset fails at every
        // single search call instead, because `VectorCollection::search`
        // refuses an unnamed field when there are several. Neither is
        // something the caller can see, so `build` refuses it here.
        let err = builder(two_vector_collection())
            .build()
            .unwrap_err()
            .to_string();
        assert!(err.contains("which one to use"), "{err}");

        let err = builder(two_vector_collection())
            .vector_field("nope")
            .build()
            .unwrap_err()
            .to_string();
        assert!(err.contains("not one of this collection's"), "{err}");
    }

    #[test]
    fn a_read_delete_only_provider_needs_no_vector_field() {
        // Neither `get` nor `delete` touches a vector — they work by key — so
        // a collection with several vectors and none named, or with none at
        // all, is perfectly serviceable for them.
        let provider = builder(two_vector_collection())
            .include_search(false)
            .build()
            .expect("get and delete need no vector field");
        let names: Vec<_> = provider.tools().iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["get", "delete"]);

        let definition = VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id").with_type("str"),
            VectorStoreField::data("text").with_type("str"),
        ])
        .unwrap();
        let store = InMemoryVectorStore::new();
        let vectorless: Arc<dyn VectorCollection> =
            Arc::from(store.get_collection("notes", definition).unwrap());
        assert!(builder(vectorless).include_search(false).build().is_ok());
    }

    #[test]
    fn asking_for_search_on_a_vectorless_collection_still_says_so() {
        let definition = VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id").with_type("str"),
            VectorStoreField::data("text").with_type("str"),
        ])
        .unwrap();
        let store = InMemoryVectorStore::new();
        let vectorless: Arc<dyn VectorCollection> =
            Arc::from(store.get_collection("notes", definition).unwrap());
        let err = builder(vectorless).build().unwrap_err().to_string();
        assert!(
            err.contains("needs a collection with a vector field"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn a_named_vector_field_reaches_the_search() {
        let collection = two_vector_collection();
        collection
            .upsert(vec![json!({
                "id": "n1",
                "text": "hello",
                "title_embedding": [1.0, 0.0, 0.0],
                "body_embedding": [0.0, 1.0, 0.0],
            })])
            .await
            .unwrap();
        let provider = builder(Arc::clone(&collection))
            .vector_field("body_embedding")
            .build()
            .unwrap();
        // Searching at all proves the field was named: an unnamed one is a
        // hard error on this collection.
        let found = call(&provider, "search", json!({ "query": "hel" })).await;
        assert_eq!(found["results"][0]["record"]["id"], json!("n1"));
    }

    #[test]
    fn a_single_vector_collection_still_needs_no_naming() {
        let provider = builder(collection()).build().unwrap();
        assert!(provider.tools().iter().any(|t| t.name == "search"));
    }

    #[test]
    fn a_provider_operator_scope_is_refused_when_a_local_tool_needs_it() {
        // `matches` errors on a provider operator, so get/delete/upsert would
        // fail on every call while search worked — a split that only shows up
        // at runtime.
        let filter: FilterExpression = Filter::new(
            "tenant",
            crate::vectors::FilterOperator::provider("azure_ai_search.match").unwrap(),
            Some(json!("acme")),
        )
        .unwrap()
        .into();
        let err = builder(collection())
            .scope_filter(filter.clone())
            .build()
            .unwrap_err()
            .to_string();
        assert!(err.contains("provider-specific operator"), "{err}");

        // A search-only provider hands the filter to the connector and never
        // evaluates it here, so it is fine.
        assert!(builder(collection())
            .scope_filter(filter)
            .include_get(false)
            .include_delete(false)
            .build()
            .is_ok());
    }

    #[test]
    fn a_vector_field_scope_is_refused_when_a_local_tool_needs_it() {
        // `get`/`delete` fetch without vectors and `upsert` checks the scope
        // before deriving one, so a missing field would read as a non-match:
        // every record silently out of scope, with no error to explain it.
        let filter: FilterExpression =
            Filter::new("embedding", crate::vectors::FilterOperator::Exists, None)
                .unwrap()
                .into();
        let err = builder(collection())
            .scope_filter(filter)
            .build()
            .unwrap_err()
            .to_string();
        assert!(err.contains("vector field"), "{err}");
    }

    #[test]
    fn an_ordinary_scope_filter_is_still_accepted() {
        assert!(builder(collection())
            .scope_filter(Filter::eq("tenant", "acme").unwrap())
            .embed_from_field("text")
            .build()
            .is_ok());
    }

    #[test]
    fn a_scope_filter_naming_an_unknown_field_is_refused_at_build() {
        // Not at the first tool call, where the caller is no longer looking.
        let err = builder(collection())
            .scope_filter(Filter::eq("nope", "x").unwrap())
            .build()
            .unwrap_err()
            .to_string();
        assert!(err.contains("does not declare"), "{err}");
    }

    #[tokio::test]
    async fn before_run_contributes_its_tools_and_instructions() {
        let provider = builder(collection()).build().unwrap();
        let mut ctx = SessionContext::new(vec![]);
        provider.before_run(&mut ctx).await.unwrap();
        assert_eq!(ctx.tools.len(), 3);
        let instructions = ctx.instructions.unwrap();
        assert!(instructions.contains("source of truth"), "{instructions}");
        assert!(instructions.contains("use search"), "{instructions}");
    }

    #[tokio::test]
    async fn upsert_derives_the_vector_and_search_finds_the_record() {
        let collection = collection();
        let provider = builder(Arc::clone(&collection))
            .embed_from_field("text")
            .build()
            .unwrap();

        let written = call(
            &provider,
            "upsert",
            json!({ "records": [{ "id": "n1", "text": "hello", "tenant": "acme" }] }),
        )
        .await;
        assert_eq!(written["keys"], json!(["n1"]));

        // The record carries a vector the model never supplied.
        let stored = collection.get(vec![json!("n1")], true).await.unwrap();
        assert_eq!(
            stored[0].as_ref().unwrap()["embedding"],
            json!([5.0, 1.0, 0.0])
        );

        let found = call(&provider, "search", json!({ "query": "hello" })).await;
        assert_eq!(found["results"][0]["record"]["id"], json!("n1"));
    }

    #[tokio::test]
    async fn a_scope_filter_hides_another_groups_records_from_every_tool() {
        let collection = collection();
        collection
            .upsert(vec![
                json!({"id": "mine", "text": "a", "tenant": "acme", "embedding": [1.0, 1.0, 0.0]}),
                json!({"id": "theirs", "text": "b", "tenant": "other", "embedding": [1.0, 1.0, 0.0]}),
            ])
            .await
            .unwrap();
        let provider = builder(Arc::clone(&collection))
            .scope_filter(Filter::eq("tenant", "acme").unwrap())
            .build()
            .unwrap();

        // Search never sees it.
        let found = call(&provider, "search", json!({ "query": "a" })).await;
        let ids: Vec<_> = found["results"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["record"]["id"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(ids, vec!["mine"]);

        // A read of it reads as absent rather than as a refusal — saying "you
        // may not read that" confirms it exists.
        let read = call(&provider, "get", json!({ "keys": ["theirs"] })).await;
        assert_eq!(read["records"][0]["record"], Value::Null);

        // And a delete of it does nothing, while reporting nothing about why.
        let deleted = call(&provider, "delete", json!({ "keys": ["theirs", "mine"] })).await;
        assert_eq!(deleted["deleted"], json!(1));
        assert_eq!(deleted["not_deleted"], json!(1));
        assert!(collection.get(vec![json!("theirs")], false).await.unwrap()[0].is_some());
    }

    #[tokio::test]
    async fn an_upsert_cannot_overwrite_another_groups_record_by_key() {
        // Laundering by key collision rather than by payload: the *payload*
        // is in scope, so the earlier check passes, but the record it lands
        // on is another group's — one `get` and `delete` refuse to touch.
        let collection = collection();
        collection
            .upsert(vec![json!({
                "id": "theirs",
                "text": "b",
                "tenant": "other",
                "embedding": [1.0, 1.0, 0.0],
            })])
            .await
            .unwrap();
        let provider = builder(Arc::clone(&collection))
            .scope_filter(Filter::eq("tenant", "acme").unwrap())
            .embed_from_field("text")
            .build()
            .unwrap();
        let tool = provider
            .tools()
            .iter()
            .find(|t| t.name == "upsert")
            .unwrap();
        let err = tool
            .executor
            .as_ref()
            .unwrap()
            .invoke(json!({
                "records": [{ "id": "theirs", "text": "mine now", "tenant": "acme" }]
            }))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("outside this tool's scope"), "{err}");

        // And the record it aimed at is untouched.
        let stored = collection.get(vec![json!("theirs")], false).await.unwrap();
        assert_eq!(stored[0].as_ref().unwrap()["tenant"], json!("other"));
    }

    #[tokio::test]
    async fn an_upsert_over_a_free_key_still_works() {
        let collection = collection();
        let provider = builder(Arc::clone(&collection))
            .scope_filter(Filter::eq("tenant", "acme").unwrap())
            .embed_from_field("text")
            .build()
            .unwrap();
        let written = call(
            &provider,
            "upsert",
            json!({ "records": [{ "id": "fresh", "text": "hello", "tenant": "acme" }] }),
        )
        .await;
        assert_eq!(written["keys"], json!(["fresh"]));
    }

    #[tokio::test]
    async fn deleting_an_absent_key_does_not_report_it_as_deleted() {
        // Without a read on the unscoped path, `deleted` was simply the input
        // length — so a key that never existed came back as a deletion, which
        // is the opposite of what the counts promise.
        let collection = collection();
        collection
            .upsert(vec![json!({
                "id": "real",
                "text": "a",
                "tenant": "acme",
                "embedding": [1.0, 1.0, 0.0],
            })])
            .await
            .unwrap();
        let provider = builder(Arc::clone(&collection)).build().unwrap();
        let result = call(&provider, "delete", json!({ "keys": ["real", "ghost"] })).await;
        assert_eq!(result["deleted"], json!(1));
        assert_eq!(result["not_deleted"], json!(1));
    }

    #[tokio::test]
    async fn a_key_named_twice_counts_once() {
        let collection = collection();
        collection
            .upsert(vec![json!({
                "id": "real",
                "text": "a",
                "tenant": "acme",
                "embedding": [1.0, 1.0, 0.0],
            })])
            .await
            .unwrap();
        let provider = builder(Arc::clone(&collection)).build().unwrap();
        let result = call(&provider, "delete", json!({ "keys": ["real", "real"] })).await;
        assert_eq!(result["deleted"], json!(1), "one record, one deletion");
        assert_eq!(result["not_deleted"], json!(0));
    }

    #[tokio::test]
    async fn an_out_of_scope_write_is_refused_rather_than_rewritten() {
        let provider = builder(collection())
            .scope_filter(Filter::eq("tenant", "acme").unwrap())
            .embed_from_field("text")
            .build()
            .unwrap();
        let tool = provider
            .tools()
            .iter()
            .find(|t| t.name == "upsert")
            .unwrap();
        let err = tool
            .executor
            .as_ref()
            .unwrap()
            .invoke(json!({ "records": [{ "id": "n1", "text": "hello", "tenant": "other" }] }))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("outside this tool's scope"), "{err}");
    }

    #[tokio::test]
    async fn a_batch_above_the_cap_is_refused() {
        let provider = builder(collection())
            .max_tool_batch_size(2)
            .build()
            .unwrap();
        let tool = provider.tools().iter().find(|t| t.name == "get").unwrap();
        let err = tool
            .executor
            .as_ref()
            .unwrap()
            .invoke(json!({ "keys": ["a", "b", "c"] }))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("limit of 2"), "{err}");
    }

    #[test]
    fn an_embedding_source_declared_non_string_is_refused_at_build() {
        // The generated schema would ask the model for an integer and the
        // executor reads the field with `as_str`, so every schema-valid
        // upsert would fail at runtime — a contradiction the caller can only
        // see here.
        let definition = VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id").with_type("str"),
            VectorStoreField::data("rank").with_type("int"),
            VectorStoreField::vector("embedding", 3),
        ])
        .unwrap();
        let store = InMemoryVectorStore::new();
        let collection: Arc<dyn VectorCollection> =
            Arc::from(store.get_collection("notes", definition).unwrap());
        let err = builder(collection)
            .embed_from_field("rank")
            .build()
            .unwrap_err()
            .to_string();
        assert!(err.contains("has to be a string"), "{err}");
    }

    #[test]
    fn an_undeclared_source_type_is_left_alone() {
        // Guessing here would reject a perfectly good untyped text field;
        // the executor's own check covers it.
        let definition = VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id").with_type("str"),
            VectorStoreField::data("text"),
            VectorStoreField::vector("embedding", 3),
        ])
        .unwrap();
        let store = InMemoryVectorStore::new();
        let collection: Arc<dyn VectorCollection> =
            Arc::from(store.get_collection("notes", definition).unwrap());
        let provider = builder(collection)
            .embed_from_field("text")
            .build()
            .expect("an untyped text field is fine");
        assert!(provider.tools().iter().any(|t| t.name == "upsert"));
    }

    #[test]
    fn the_key_schema_follows_the_key_fields_declared_type() {
        // A collection keyed by an integer stores and looks up the JSON
        // number `42`; `InMemoryVectorStore` keys its map on
        // `Value::to_string()`, so a model told `string` sends `"42"`, which
        // does not collide with `42` — it simply never matches, and every
        // read and delete silently finds nothing.
        let definition = VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id").with_type("int"),
            VectorStoreField::data("text").with_type("str"),
            VectorStoreField::vector("embedding", 3),
        ])
        .unwrap();
        let store = InMemoryVectorStore::new();
        let collection: Arc<dyn VectorCollection> =
            Arc::from(store.get_collection("notes", definition).unwrap());
        let provider = builder(collection).build().unwrap();
        for name in ["get", "delete"] {
            let tool = provider.tools().iter().find(|t| t.name == name).unwrap();
            assert_eq!(
                tool.parameters["properties"]["keys"]["items"]["type"],
                json!("integer"),
                "{name}"
            );
        }
    }

    #[test]
    fn a_string_key_still_reads_as_a_string() {
        let provider = builder(collection()).build().unwrap();
        let tool = provider.tools().iter().find(|t| t.name == "get").unwrap();
        assert_eq!(
            tool.parameters["properties"]["keys"]["items"]["type"],
            json!("string")
        );
    }

    #[test]
    fn the_vector_field_is_never_in_the_upsert_schema() {
        let provider = builder(collection())
            .embed_from_field("text")
            .build()
            .unwrap();
        let tool = provider
            .tools()
            .iter()
            .find(|t| t.name == "upsert")
            .unwrap();
        let properties = &tool.parameters["properties"]["records"]["items"]["properties"];
        assert!(properties.get("embedding").is_none());
        assert!(properties.get("text").is_some());
    }
}
