//! # agent-framework-azure-ai-search
//!
//! An Azure AI Search [`ContextProvider`] for `agent-framework-rs`: it runs a
//! hybrid/semantic search against a search index and injects the retrieved
//! documents into an agent invocation as extra context.
//!
//! It talks the Azure AI Search REST API directly (no Azure SDK dependency):
//! `POST {endpoint}/indexes('{index}')/docs/search?api-version=2024-07-01`,
//! authenticating with either an admin/query **api-key** or a Microsoft Entra
//! ID **bearer token** (via a [`TokenCredential`]). This is the "semantic"
//! retrieval mode of the upstream Python provider — fast hybrid search with an
//! optional semantic reranker and optional vector query. (The Python provider's
//! separate "agentic"/Knowledge-Base mode is out of scope here.)
//!
//! ```no_run
//! use std::sync::Arc;
//! use agent_framework_azure_ai_search::AzureAISearchProvider;
//! use agent_framework_core::memory::ContextProvider;
//! use agent_framework_core::types::ChatMessage;
//!
//! # async fn demo() -> agent_framework_core::error::Result<()> {
//! let provider = AzureAISearchProvider::with_api_key(
//!     "https://my-search.search.windows.net",
//!     "my-index",
//!     "my-query-key",
//! )
//! .with_top(5)
//! .with_semantic_configuration("my-semantic-config");
//!
//! let context = provider
//!     .invoking(&[ChatMessage::user("What is in the documents?")])
//!     .await?;
//! println!("{}", context.instructions.unwrap_or_default());
//! # Ok(())
//! # }
//! ```

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use agent_framework_azure::TokenCredential;
use agent_framework_core::error::{Error, Result};
use agent_framework_core::memory::{Context, ContextProvider};
use agent_framework_core::types::{ChatMessage, Role};
use async_trait::async_trait;
use serde_json::{json, Map, Value};

/// The default Azure AI Search REST API version.
pub const DEFAULT_API_VERSION: &str = "2024-07-01";

/// The Entra ID scope (audience) for the Azure AI Search data plane.
pub const SEARCH_SCOPE: &str = "https://search.azure.com/.default";

/// The default prompt prepended to retrieved context, matching the upstream
/// Python provider.
pub const DEFAULT_CONTEXT_PROMPT: &str = "Use the following context to answer the question:";

type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;
type EmbeddingFn = Arc<dyn Fn(&str) -> BoxFuture<Result<Vec<f32>>> + Send + Sync>;

/// How a search request authenticates.
#[derive(Clone)]
enum SearchAuth {
    /// `api-key: <key>` header.
    ApiKey(String),
    /// `Authorization: Bearer <token>` from a [`TokenCredential`].
    Credential(Arc<dyn TokenCredential>),
}

/// A context provider backed by an Azure AI Search index.
#[derive(Clone)]
pub struct AzureAISearchProvider {
    http: reqwest::Client,
    endpoint: String,
    index_name: String,
    api_version: String,
    scope: String,
    auth: SearchAuth,
    top_k: usize,
    select_fields: Option<Vec<String>>,
    semantic_configuration_name: Option<String>,
    vector_field_name: Option<String>,
    embedding_function: Option<EmbeddingFn>,
    context_prompt: String,
}

impl std::fmt::Debug for AzureAISearchProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AzureAISearchProvider")
            .field("endpoint", &self.endpoint)
            .field("index_name", &self.index_name)
            .field("api_version", &self.api_version)
            .field("top_k", &self.top_k)
            .field(
                "semantic_configuration_name",
                &self.semantic_configuration_name,
            )
            .field("vector_field_name", &self.vector_field_name)
            .field(
                "auth",
                &match &self.auth {
                    SearchAuth::ApiKey(_) => "api-key",
                    SearchAuth::Credential(_) => "token-credential",
                },
            )
            .finish_non_exhaustive()
    }
}

impl AzureAISearchProvider {
    fn build(endpoint: String, index_name: String, auth: SearchAuth) -> Self {
        Self {
            http: reqwest::Client::new(),
            endpoint,
            index_name,
            api_version: DEFAULT_API_VERSION.to_string(),
            scope: SEARCH_SCOPE.to_string(),
            auth,
            top_k: 5,
            select_fields: None,
            semantic_configuration_name: None,
            vector_field_name: None,
            embedding_function: None,
            context_prompt: DEFAULT_CONTEXT_PROMPT.to_string(),
        }
    }

    /// Create a provider authenticating with a search **api-key**
    /// (`api-key` header).
    pub fn with_api_key(
        endpoint: impl Into<String>,
        index_name: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self::build(
            endpoint.into(),
            index_name.into(),
            SearchAuth::ApiKey(api_key.into()),
        )
    }

    /// Create a provider authenticating via a [`TokenCredential`]
    /// (`Authorization: Bearer <token>`, scope [`SEARCH_SCOPE`]).
    pub fn with_token_credential(
        endpoint: impl Into<String>,
        index_name: impl Into<String>,
        credential: Arc<dyn TokenCredential>,
    ) -> Self {
        Self::build(
            endpoint.into(),
            index_name.into(),
            SearchAuth::Credential(credential),
        )
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

    /// Set the maximum number of documents to retrieve (default `5`).
    pub fn with_top(mut self, top: usize) -> Self {
        self.top_k = top;
        self
    }

    /// Restrict the fields returned for each document (`$select`).
    pub fn with_select_fields<I, S>(mut self, fields: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.select_fields = Some(fields.into_iter().map(Into::into).collect());
        self
    }

    /// Enable semantic ranking with the named semantic configuration.
    pub fn with_semantic_configuration(mut self, name: impl Into<String>) -> Self {
        self.semantic_configuration_name = Some(name.into());
        self
    }

    /// Add a vector query over `field`. Without an embedding function the query
    /// text is sent for server-side vectorization (`{"kind":"text",…}`); with
    /// one (see [`with_embedding_function`](Self::with_embedding_function)) the
    /// computed vector is sent (`{"kind":"vector",…}`).
    pub fn with_vector_field(mut self, field: impl Into<String>) -> Self {
        self.vector_field_name = Some(field.into());
        self
    }

    /// Supply a client-side embedding function used to vectorize the query for
    /// the configured [`vector field`](Self::with_vector_field).
    pub fn with_embedding_function<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Vec<f32>>> + Send + 'static,
    {
        self.embedding_function = Some(Arc::new(move |q: &str| Box::pin(f(q.to_string()))));
        self
    }

    /// Override the prompt prepended to retrieved context
    /// (default [`DEFAULT_CONTEXT_PROMPT`]).
    pub fn with_context_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.context_prompt = prompt.into();
        self
    }

    fn search_url(&self) -> String {
        format!(
            "{}/indexes('{}')/docs/search?api-version={}",
            self.endpoint.trim_end_matches('/'),
            self.index_name,
            self.api_version
        )
    }

    /// The `k` for a vector query: a larger neighborhood improves reranking
    /// quality when a semantic configuration is set (mirrors the Python
    /// provider).
    fn vector_k(&self) -> usize {
        if self.semantic_configuration_name.is_some() {
            self.top_k.max(50)
        } else {
            self.top_k
        }
    }

    /// Build the `docs/search` request body for `query`.
    async fn build_search_body(&self, query: &str) -> Result<Value> {
        let mut body = Map::new();
        body.insert("search".into(), json!(query));
        body.insert("top".into(), json!(self.top_k));
        if let Some(fields) = &self.select_fields {
            body.insert("select".into(), json!(fields.join(",")));
        }
        if let Some(sem) = &self.semantic_configuration_name {
            body.insert("queryType".into(), json!("semantic"));
            body.insert("semanticConfiguration".into(), json!(sem));
            body.insert("captions".into(), json!("extractive"));
        }
        if let Some(field) = &self.vector_field_name {
            let vector_query = match &self.embedding_function {
                Some(embed) => {
                    let vector = embed(query).await?;
                    json!({"kind": "vector", "vector": vector, "fields": field, "k": self.vector_k()})
                }
                None => {
                    json!({"kind": "text", "text": query, "fields": field, "k": self.vector_k()})
                }
            };
            body.insert("vectorQueries".into(), json!([vector_query]));
        }
        Ok(Value::Object(body))
    }

    /// Run the search and return the formatted document strings.
    async fn search(&self, query: &str) -> Result<Vec<String>> {
        let body = self.build_search_body(query).await?;
        let mut req = self.http.post(self.search_url()).json(&body);
        req = match &self.auth {
            SearchAuth::ApiKey(key) => req.header("api-key", key),
            SearchAuth::Credential(cred) => {
                let token = cred.get_token_for_scope(&self.scope).await?;
                req.bearer_auth(token)
            }
        };
        let resp = req
            .send()
            .await
            .map_err(|e| Error::service(format!("request failed: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(Error::service_status(
                status.as_u16(),
                format!("Azure AI Search error {status}: {text}"),
                None,
            ));
        }
        let value: Value = resp
            .json()
            .await
            .map_err(|e| Error::service(format!("invalid response json: {e}")))?;
        Ok(parse_search_results(&value))
    }
}

/// The text of the most recent non-empty user message — the retrieval query.
fn latest_user_query(messages: &[ChatMessage]) -> Option<String> {
    messages.iter().rev().find_map(|m| {
        if m.role == Role::user() {
            let text = m.text();
            if !text.trim().is_empty() {
                return Some(text);
            }
        }
        None
    })
}

/// Extract the formatted, cited document strings from a `docs/search` response.
fn parse_search_results(value: &Value) -> Vec<String> {
    let Some(docs) = value.get("value").and_then(Value::as_array) else {
        return Vec::new();
    };
    docs.iter()
        .filter_map(|doc| {
            let text = extract_document_text(doc);
            if text.is_empty() {
                None
            } else {
                Some(text)
            }
        })
        .collect()
}

/// Extract readable text from a search document, with a `[Source: <id>]`
/// citation prefix. Mirrors the Python provider: try common text fields, else
/// concatenate all string fields (excluding `@…`/`id`).
fn extract_document_text(doc: &Value) -> String {
    let Some(obj) = doc.as_object() else {
        return String::new();
    };

    let mut text = String::new();
    for field in ["content", "text", "description", "body", "chunk"] {
        if let Some(v) = obj.get(field).and_then(Value::as_str) {
            text = v.to_string();
            break;
        }
    }
    if text.is_empty() {
        let parts: Vec<String> = obj
            .iter()
            .filter_map(|(k, v)| match v {
                Value::String(s) if !k.starts_with('@') && k != "id" => Some(format!("{k}: {s}")),
                _ => None,
            })
            .collect();
        text = parts.join(" | ");
    }

    let doc_id = obj
        .get("id")
        .and_then(Value::as_str)
        .or_else(|| obj.get("@search.id").and_then(Value::as_str));
    match doc_id {
        Some(id) if !text.is_empty() => format!("[Source: {id}] {text}"),
        _ => text,
    }
}

#[async_trait]
impl ContextProvider for AzureAISearchProvider {
    async fn invoking(&self, messages: &[ChatMessage]) -> Result<Context> {
        let Some(query) = latest_user_query(messages) else {
            return Ok(Context::new());
        };
        let results = self.search(&query).await?;
        if results.is_empty() {
            return Ok(Context::new());
        }
        // Fold the header + one block per result into the injected instructions.
        let mut instructions = self.context_prompt.clone();
        for part in &results {
            instructions.push('\n');
            instructions.push_str(part);
        }
        Ok(Context::new().with_instructions(instructions))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider() -> AzureAISearchProvider {
        AzureAISearchProvider::with_api_key("https://s.search.windows.net", "idx", "key")
    }

    #[test]
    fn search_url_uses_odata_index_path() {
        assert_eq!(
            provider().search_url(),
            "https://s.search.windows.net/indexes('idx')/docs/search?api-version=2024-07-01"
        );
    }

    #[tokio::test]
    async fn body_includes_semantic_and_select() {
        let p = provider()
            .with_top(3)
            .with_select_fields(["id", "content"])
            .with_semantic_configuration("sem-cfg");
        let body = p.build_search_body("hello").await.unwrap();
        assert_eq!(body["search"], json!("hello"));
        assert_eq!(body["top"], json!(3));
        assert_eq!(body["select"], json!("id,content"));
        assert_eq!(body["queryType"], json!("semantic"));
        assert_eq!(body["semanticConfiguration"], json!("sem-cfg"));
        assert_eq!(body["captions"], json!("extractive"));
    }

    #[tokio::test]
    async fn body_server_side_vector_query_without_embedding() {
        let p = provider().with_vector_field("embedding");
        let body = p.build_search_body("hi").await.unwrap();
        let vq = &body["vectorQueries"][0];
        assert_eq!(vq["kind"], json!("text"));
        assert_eq!(vq["text"], json!("hi"));
        assert_eq!(vq["fields"], json!("embedding"));
        assert_eq!(vq["k"], json!(5));
    }

    #[tokio::test]
    async fn body_client_side_vector_query_with_embedding() {
        let p = provider()
            .with_semantic_configuration("sem")
            .with_vector_field("embedding")
            .with_embedding_function(|_q| async { Ok(vec![0.1, 0.2, 0.3]) });
        let body = p.build_search_body("hi").await.unwrap();
        let vq = &body["vectorQueries"][0];
        assert_eq!(vq["kind"], json!("vector"));
        // The computed embedding is passed through (f32 precision).
        assert_eq!(vq["vector"].as_array().unwrap().len(), 3);
        assert!(vq.get("text").is_none());
        // With a semantic config the vector `k` widens to at least 50.
        assert_eq!(vq["k"], json!(50));
    }

    #[test]
    fn document_text_uses_content_and_citation() {
        let doc = json!({"id": "doc123", "content": "Test document content"});
        assert_eq!(
            extract_document_text(&doc),
            "[Source: doc123] Test document content"
        );
    }

    #[test]
    fn document_text_falls_back_to_string_fields() {
        let doc = json!({"title": "T", "body_field": "B", "@search.score": 1.2});
        let text = extract_document_text(&doc);
        assert!(text.contains("title: T"), "got: {text}");
        assert!(text.contains("body_field: B"), "got: {text}");
        assert!(!text.contains("@search.score"), "got: {text}");
    }

    #[test]
    fn parse_results_skips_empty_docs() {
        let value = json!({"value": [
            {"id": "d1", "content": "hello"},
            {"id": "d2"},
        ]});
        let results = parse_search_results(&value);
        assert_eq!(results, vec!["[Source: d1] hello".to_string()]);
    }

    #[test]
    fn latest_user_query_picks_last_nonempty_user() {
        let msgs = vec![
            ChatMessage::user("first"),
            ChatMessage::assistant("reply"),
            ChatMessage::user("second"),
        ];
        assert_eq!(latest_user_query(&msgs).as_deref(), Some("second"));

        // No user text → no query.
        assert_eq!(latest_user_query(&[ChatMessage::system("sys")]), None);
        assert_eq!(latest_user_query(&[ChatMessage::user("   ")]), None);
    }
}
