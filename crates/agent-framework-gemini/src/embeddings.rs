//! Google Gemini embeddings client (Gemini Embedding 2).
//!
//! Speaks the `batchEmbedContents` REST route directly
//! (`POST {base}/v1beta/models/{model}:batchEmbedContents`), matching this
//! crate's chat client rather than taking on the `google-genai` SDK.
//!
//! # The task prefix is the whole point
//!
//! Gemini Embedding 2 conditions a vector on what the text is *for*, and it
//! takes that from a prefix on the text itself — not from a request field.
//! Upstream's client sends `output_dimensionality` as the only config and
//! rewrites each string (`agent_framework_gemini/_embedding_client.py`), and
//! so does this one. A `taskType` request field exists on the older
//! `text-embedding-004` and `gemini-embedding-001` routes; sending it here
//! would do nothing, and skipping the prefix produces a vector conditioned on
//! nothing in particular — measurably worse at the job it was asked for,
//! while still looking like a perfectly good embedding.
//!
//! Because of that, a task is **required** for every call, as upstream
//! requires it. There is no safe default: [`GeminiEmbeddingTask::RetrievalDocument`]
//! on a search query would index the query as a document, and no prefix at
//! all silently opts out of the model's conditioning. Both failures are
//! invisible in the response and show up only as worse retrieval, so this
//! client refuses the call instead.
//!
//! ```no_run
//! use agent_framework_core::client::EmbeddingClient;
//! use agent_framework_gemini::{GeminiEmbeddingClient, GeminiEmbeddingTask, GeminiEmbeddingOptions};
//! use agent_framework_core::types::EmbeddingGenerationOptions;
//!
//! # async fn demo() -> agent_framework_core::error::Result<()> {
//! let client = GeminiEmbeddingClient::from_env()?;
//!
//! // Indexing: a document, optionally with a title.
//! let docs = client
//!     .get_embeddings(
//!         vec!["The Rust book, chapter 4".into()],
//!         Some(
//!             EmbeddingGenerationOptions::new()
//!                 .with_task(GeminiEmbeddingTask::RetrievalDocument)
//!                 .with_title("Ownership"),
//!         ),
//!     )
//!     .await?;
//!
//! // Searching: the *same* text would get a different vector here, which is
//! // the conditioning doing its job.
//! let query = client
//!     .get_embeddings(
//!         vec!["who owns a value".into()],
//!         Some(EmbeddingGenerationOptions::new().with_task(GeminiEmbeddingTask::RetrievalQuery)),
//!     )
//!     .await?;
//! # let _ = (docs, query);
//! # Ok(())
//! # }
//! ```
//!
//! # Scope
//!
//! Text only. Upstream also accepts `google.genai` `Content`/`Part` values to
//! embed images and other media, which this port's
//! [`EmbeddingClient`] trait
//! cannot express — it takes `Vec<String>`. Widening the trait is a core
//! change affecting every embedding client, so it is recorded as a gap rather
//! than forced through a provider-shaped side door.

use std::collections::HashMap;
use std::sync::Arc;

use agent_framework_core::client::EmbeddingClient;
use agent_framework_core::error::{Error, Result};
use agent_framework_core::types::{Embedding, EmbeddingGenerationOptions, GeneratedEmbeddings};
use serde_json::{json, Map, Value};

use crate::{classify_gemini_error, parse_retry_after, API_VERSION, DEFAULT_BASE_URL};

/// The default (and stable) Gemini embedding model.
pub const DEFAULT_EMBEDDING_MODEL: &str = "gemini-embedding-2";

/// The models this client will embed against.
///
/// An allowlist rather than a free-form string, as upstream has: the task
/// prefixing below *is* Embedding 2's calling convention, so applying it to
/// `gemini-embedding-001` or `text-embedding-004` would prefix text those
/// models take literally — a silent accuracy loss, since the prefix becomes
/// part of what gets embedded.
pub const SUPPORTED_EMBEDDING_MODELS: [&str; 2] =
    ["gemini-embedding-2", "gemini-embedding-2-preview"];

/// The `additional_properties` key carrying the task, set by
/// [`GeminiEmbeddingOptions::with_task`].
const TASK_PROPERTY: &str = "task_type";
/// The `additional_properties` key carrying a document title, set by
/// [`GeminiEmbeddingOptions::with_title`].
const TITLE_PROPERTY: &str = "title";

/// What an embedding is *for*, which Gemini Embedding 2 conditions the vector
/// on. See the [module docs](self).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeminiEmbeddingTask {
    /// Text being indexed for later retrieval. The only task that accepts a
    /// [`title`](GeminiEmbeddingOptions::with_title).
    RetrievalDocument,
    /// A search query against indexed documents.
    RetrievalQuery,
    /// A question, for retrieving passages that answer it.
    QuestionAnswering,
    /// A claim, for retrieving evidence that supports or refutes it.
    FactVerification,
    /// A query over code.
    CodeRetrievalQuery,
    /// Text being assigned to a label.
    Classification,
    /// Text being grouped by similarity.
    Clustering,
    /// Text being compared with other text for likeness.
    SemanticSimilarity,
}

impl GeminiEmbeddingTask {
    /// The wire name, which is also what upstream's `task_type` option takes.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RetrievalDocument => "RETRIEVAL_DOCUMENT",
            Self::RetrievalQuery => "RETRIEVAL_QUERY",
            Self::QuestionAnswering => "QUESTION_ANSWERING",
            Self::FactVerification => "FACT_VERIFICATION",
            Self::CodeRetrievalQuery => "CODE_RETRIEVAL_QUERY",
            Self::Classification => "CLASSIFICATION",
            Self::Clustering => "CLUSTERING",
            Self::SemanticSimilarity => "SEMANTIC_SIMILARITY",
        }
    }

    /// Parse a wire name, for an option set as a raw string (as a generic
    /// caller or a deserialized config would).
    pub fn parse(name: &str) -> Option<Self> {
        [
            Self::RetrievalDocument,
            Self::RetrievalQuery,
            Self::QuestionAnswering,
            Self::FactVerification,
            Self::CodeRetrievalQuery,
            Self::Classification,
            Self::Clustering,
            Self::SemanticSimilarity,
        ]
        .into_iter()
        .find(|t| t.as_str().eq_ignore_ascii_case(name))
    }

    /// The phrase this task contributes to a query prefix, or `None` for
    /// [`RetrievalDocument`](Self::RetrievalDocument), which uses the
    /// title/text form instead.
    fn query_phrase(self) -> Option<&'static str> {
        Some(match self {
            Self::RetrievalDocument => return None,
            Self::RetrievalQuery => "search result",
            Self::QuestionAnswering => "question answering",
            Self::FactVerification => "fact checking",
            Self::CodeRetrievalQuery => "code retrieval",
            Self::Classification => "classification",
            Self::Clustering => "clustering",
            Self::SemanticSimilarity => "sentence similarity",
        })
    }

    /// Condition `text` on this task, per the Embedding 2 task instructions.
    fn prepare(self, text: &str, title: Option<&str>) -> String {
        match self.query_phrase() {
            // A document carries its title, and literally the word "none"
            // when it has none — the model is conditioned on the field being
            // present, so omitting it would be a different prompt.
            None => format!("title: {} | text: {text}", title.unwrap_or("none")),
            Some(phrase) => format!("task: {phrase} | query: {text}"),
        }
    }
}

/// Gemini-specific embedding options, set on the core
/// [`EmbeddingGenerationOptions`].
///
/// They live in `additional_properties` rather than as typed fields because
/// that struct is shared by every provider; this trait is the typed way to
/// write them, so a caller does not have to know the key names or the wire
/// spelling of a task.
pub trait GeminiEmbeddingOptions: Sized {
    /// Set what these embeddings are for. Required — see the
    /// [module docs](self).
    fn with_task(self, task: GeminiEmbeddingTask) -> Self;

    /// Set the title of the document being embedded. Only meaningful with
    /// [`GeminiEmbeddingTask::RetrievalDocument`]; the client rejects the call
    /// otherwise rather than drop it, since a title set against a query task
    /// means the caller believed something untrue about the request.
    fn with_title(self, title: impl Into<String>) -> Self;
}

impl GeminiEmbeddingOptions for EmbeddingGenerationOptions {
    fn with_task(mut self, task: GeminiEmbeddingTask) -> Self {
        self.additional_properties
            .insert(TASK_PROPERTY.to_string(), json!(task.as_str()));
        self
    }

    fn with_title(mut self, title: impl Into<String>) -> Self {
        self.additional_properties
            .insert(TITLE_PROPERTY.to_string(), json!(title.into()));
        self
    }
}

/// A Google Gemini embeddings client (`batchEmbedContents`).
///
/// See the [module docs](self) for the task-prefix requirement.
#[derive(Clone)]
pub struct GeminiEmbeddingClient {
    inner: Arc<Inner>,
}

#[derive(Clone)]
struct Inner {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
}

impl std::fmt::Debug for GeminiEmbeddingClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeminiEmbeddingClient")
            .field("base_url", &self.inner.base_url)
            .field("model", &self.inner.model)
            .finish_non_exhaustive()
    }
}

impl GeminiEmbeddingClient {
    /// Create a client for `api_key`, embedding against
    /// [`DEFAULT_EMBEDDING_MODEL`].
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(Inner {
                http: reqwest::Client::new(),
                api_key: api_key.into(),
                base_url: DEFAULT_BASE_URL.to_string(),
                model: DEFAULT_EMBEDDING_MODEL.to_string(),
            }),
        }
    }

    /// Build from the environment: `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) and
    /// the optional `GOOGLE_EMBEDDING_MODEL` override, the same variable
    /// upstream reads.
    ///
    /// # Errors
    /// [`Error::Configuration`] when no API key is set, or when
    /// `GOOGLE_EMBEDDING_MODEL` names a model outside
    /// [`SUPPORTED_EMBEDDING_MODELS`].
    pub fn from_env() -> Result<Self> {
        Self::from_env_vars(|key| std::env::var(key).ok())
    }

    /// [`from_env`](Self::from_env) over an injectable lookup, so the parsing
    /// is testable without touching process environment variables.
    fn from_env_vars(get: impl Fn(&str) -> Option<String>) -> Result<Self> {
        let api_key = get("GEMINI_API_KEY")
            .or_else(|| get("GOOGLE_API_KEY"))
            .ok_or_else(|| {
                Error::Configuration("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set".into())
            })?;
        let client = Self::new(api_key);
        match get("GOOGLE_EMBEDDING_MODEL") {
            Some(model) => client.try_with_model(model),
            None => Ok(client),
        }
    }

    /// Override the embedding model.
    ///
    /// # Errors
    /// [`Error::Configuration`] when `model` is not in
    /// [`SUPPORTED_EMBEDDING_MODELS`] — see its documentation for why this is
    /// an allowlist.
    pub fn try_with_model(mut self, model: impl Into<String>) -> Result<Self> {
        let model = model.into();
        if !SUPPORTED_EMBEDDING_MODELS.contains(&model.as_str()) {
            return Err(Error::Configuration(format!(
                "unsupported Gemini embedding model '{model}'; this client speaks Embedding 2's \
                 task-prefix convention, so it accepts only {}",
                SUPPORTED_EMBEDDING_MODELS.join(" or ")
            )));
        }
        Arc::make_mut(&mut self.inner).model = model;
        Ok(self)
    }

    /// Override the API base URL (no `/v1beta` segment; it is appended per
    /// request). Mainly for tests and proxies.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).base_url = base_url.into();
        self
    }

    /// The embedding model this client targets.
    pub fn model_name(&self) -> &str {
        &self.inner.model
    }

    /// Build the `batchEmbedContents` request body, applying the task prefix
    /// to every value.
    fn build_body(
        &self,
        values: &[String],
        options: Option<&EmbeddingGenerationOptions>,
    ) -> Result<(Value, String)> {
        let empty = HashMap::new();
        let extras = options.map(|o| &o.additional_properties).unwrap_or(&empty);

        let model = match options.and_then(|o| o.model.as_deref()) {
            Some(model) => {
                if !SUPPORTED_EMBEDDING_MODELS.contains(&model) {
                    return Err(Error::Configuration(format!(
                        "unsupported Gemini embedding model '{model}'; this client accepts only {}",
                        SUPPORTED_EMBEDDING_MODELS.join(" or ")
                    )));
                }
                model.to_string()
            }
            None => self.inner.model.clone(),
        };

        let task = match extras.get(TASK_PROPERTY) {
            Some(Value::String(name)) => GeminiEmbeddingTask::parse(name).ok_or_else(|| {
                Error::Configuration(format!("unsupported Gemini embedding task '{name}'"))
            })?,
            Some(other) => {
                return Err(Error::Configuration(format!(
                "the '{TASK_PROPERTY}' embedding option must be a task name string, found {other}"
            )))
            }
            // Refused rather than defaulted: see the module docs. The message
            // names the typed setter, since the key is an implementation
            // detail of it.
            None => {
                return Err(Error::Configuration(format!(
                    "Gemini Embedding 2 conditions a vector on what the text is for, so a task is \
                     required: set one with `EmbeddingGenerationOptions::with_task` (one of {})",
                    SUPPORTED_TASK_NAMES.join(", ")
                )))
            }
        };

        let title = match extras.get(TITLE_PROPERTY) {
            Some(Value::String(t)) => Some(t.as_str()),
            Some(other) => {
                return Err(Error::Configuration(format!(
                    "the '{TITLE_PROPERTY}' embedding option must be a string, found {other}"
                )))
            }
            None => None,
        };
        if title.is_some() && task != GeminiEmbeddingTask::RetrievalDocument {
            return Err(Error::Configuration(format!(
                "a title only applies when indexing a document, but the task is {}; drop the \
                 title or use GeminiEmbeddingTask::RetrievalDocument",
                task.as_str()
            )));
        }

        let dimensions = match options.and_then(|o| o.dimensions) {
            // Zero would be accepted by the type and rejected by the service,
            // so it is named here instead.
            Some(0) => {
                return Err(Error::Configuration(
                    "dimensions must be a positive integer".into(),
                ))
            }
            other => other,
        };

        let requests: Vec<Value> = values
            .iter()
            .map(|text| {
                let mut req = Map::new();
                // `batchEmbedContents` requires the model on each entry, in
                // the resource form.
                req.insert("model".into(), json!(format!("models/{model}")));
                req.insert(
                    "content".into(),
                    json!({ "parts": [{ "text": task.prepare(text, title) }] }),
                );
                if let Some(d) = dimensions {
                    req.insert("outputDimensionality".into(), json!(d));
                }
                Value::Object(req)
            })
            .collect();

        Ok((json!({ "requests": requests }), model))
    }

    async fn post(&self, model: &str, body: &Value) -> Result<Value> {
        let url = format!(
            "{}/{API_VERSION}/models/{model}:batchEmbedContents",
            self.inner.base_url.trim_end_matches('/')
        );
        let resp = self
            .inner
            .http
            .post(&url)
            .header("x-goog-api-key", &self.inner.api_key)
            .header("content-type", "application/json")
            .json(body)
            .send()
            .await
            .map_err(|e| Error::service(format!("request failed: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let retry_after = parse_retry_after(resp.headers());
            let text = resp.text().await.unwrap_or_default();
            return Err(classify_gemini_error(
                status.as_u16(),
                &text,
                format!("Gemini API error {status}: {text}"),
                retry_after,
            ));
        }
        resp.json()
            .await
            .map_err(|e| Error::service(format!("invalid embeddings response: {e}")))
    }
}

/// Every task's wire name, for the error that asks for one.
const SUPPORTED_TASK_NAMES: [&str; 8] = [
    "RETRIEVAL_DOCUMENT",
    "RETRIEVAL_QUERY",
    "QUESTION_ANSWERING",
    "FACT_VERIFICATION",
    "CODE_RETRIEVAL_QUERY",
    "CLASSIFICATION",
    "CLUSTERING",
    "SEMANTIC_SIMILARITY",
];

/// Parse a `batchEmbedContents` response into vectors, in request order.
///
/// A count that does not match the request is an error rather than a short
/// result: the caller pairs these with its own inputs by position, so a
/// silently shorter list would misalign every vector after the gap.
pub fn parse_embeddings_response(
    value: &Value,
    expected: usize,
    model: &str,
) -> Result<GeneratedEmbeddings> {
    let items = value
        .get("embeddings")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::service("embeddings response has no `embeddings` array"))?;
    if items.len() != expected {
        return Err(Error::service(format!(
            "embeddings response has {} vectors for {expected} inputs",
            items.len()
        )));
    }
    let mut out = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
        let values = item
            .get("values")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::service(format!("embedding {i} has no `values` array")))?;
        let mut vector = Vec::with_capacity(values.len());
        for v in values {
            vector.push(v.as_f64().ok_or_else(|| {
                Error::service(format!("embedding {i} has a non-numeric component"))
            })? as f32);
        }
        out.push(Embedding {
            vector,
            model: Some(model.to_string()),
        });
    }
    Ok(GeneratedEmbeddings::new(out))
}

#[async_trait::async_trait]
impl EmbeddingClient for GeminiEmbeddingClient {
    async fn get_embeddings(
        &self,
        values: Vec<String>,
        options: Option<EmbeddingGenerationOptions>,
    ) -> Result<GeneratedEmbeddings> {
        if values.is_empty() {
            return Ok(GeneratedEmbeddings::default());
        }
        let (body, model) = self.build_body(&values, options.as_ref())?;
        let value = self.post(&model, &body).await?;
        parse_embeddings_response(&value, values.len(), &model)
    }

    fn model(&self) -> Option<&str> {
        Some(&self.inner.model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn client() -> GeminiEmbeddingClient {
        GeminiEmbeddingClient::new("test-key")
    }

    fn doc_options() -> EmbeddingGenerationOptions {
        EmbeddingGenerationOptions::new().with_task(GeminiEmbeddingTask::RetrievalDocument)
    }

    fn text_of(body: &Value, i: usize) -> String {
        body["requests"][i]["content"]["parts"][0]["text"]
            .as_str()
            .unwrap()
            .to_string()
    }

    // region: the task prefix (the point of Embedding 2)

    #[test]
    fn a_document_is_prefixed_with_its_title_or_the_literal_none() {
        let (body, _) = client()
            .build_body(
                &["chapter 4".to_string()],
                Some(&doc_options().with_title("Ownership")),
            )
            .unwrap();
        assert_eq!(text_of(&body, 0), "title: Ownership | text: chapter 4");

        // No title still sends the field, with the literal word "none": the
        // model is conditioned on the shape, so omitting it is a different
        // prompt rather than a cleaner one.
        let (body, _) = client()
            .build_body(&["chapter 4".to_string()], Some(&doc_options()))
            .unwrap();
        assert_eq!(text_of(&body, 0), "title: none | text: chapter 4");
    }

    #[test]
    fn every_query_task_has_its_own_phrase() {
        // The phrases are the model's, not ours — a wrong one conditions the
        // vector on the wrong job and shows up only as worse retrieval, so
        // each is pinned rather than spot-checked.
        for (task, expected) in [
            (GeminiEmbeddingTask::RetrievalQuery, "search result"),
            (GeminiEmbeddingTask::QuestionAnswering, "question answering"),
            (GeminiEmbeddingTask::FactVerification, "fact checking"),
            (GeminiEmbeddingTask::CodeRetrievalQuery, "code retrieval"),
            (GeminiEmbeddingTask::Classification, "classification"),
            (GeminiEmbeddingTask::Clustering, "clustering"),
            (
                GeminiEmbeddingTask::SemanticSimilarity,
                "sentence similarity",
            ),
        ] {
            let options = EmbeddingGenerationOptions::new().with_task(task);
            let (body, _) = client()
                .build_body(&["who owns it".to_string()], Some(&options))
                .unwrap();
            assert_eq!(
                text_of(&body, 0),
                format!("task: {expected} | query: who owns it"),
                "{task:?}"
            );
        }
    }

    #[test]
    fn the_same_text_gets_a_different_prompt_per_task() {
        // The whole reason the task is required: indexing and searching the
        // same string must not produce the same request.
        let (doc, _) = client()
            .build_body(&["ownership".to_string()], Some(&doc_options()))
            .unwrap();
        let (query, _) = client()
            .build_body(
                &["ownership".to_string()],
                Some(
                    &EmbeddingGenerationOptions::new()
                        .with_task(GeminiEmbeddingTask::RetrievalQuery),
                ),
            )
            .unwrap();
        assert_ne!(text_of(&doc, 0), text_of(&query, 0));
    }

    #[test]
    fn a_missing_task_is_refused_rather_than_defaulted() {
        // No default is safe: RETRIEVAL_DOCUMENT would index a query as a
        // document, and no prefix silently opts out of the conditioning.
        // Both look like success in the response.
        let err = client()
            .build_body(
                &["hi".to_string()],
                Some(&EmbeddingGenerationOptions::new()),
            )
            .expect_err("a task is required");
        let msg = err.to_string();
        assert!(msg.contains("a task is required"), "{msg}");
        // Names the typed setter, since the carrier key is its detail.
        assert!(msg.contains("with_task"), "{msg}");
        // Options omitted entirely fails the same way, which is the path a
        // generic `EmbeddingClient` caller takes.
        assert!(client().build_body(&["hi".to_string()], None).is_err());
    }

    #[test]
    fn a_task_set_as_a_raw_string_is_accepted_case_insensitively() {
        // A deserialized config or a generic caller writes the key directly;
        // the wire name is the same one upstream's option takes.
        let mut options = EmbeddingGenerationOptions::new();
        options
            .additional_properties
            .insert("task_type".into(), json!("retrieval_query"));
        let (body, _) = client()
            .build_body(&["q".to_string()], Some(&options))
            .unwrap();
        assert_eq!(text_of(&body, 0), "task: search result | query: q");

        let mut options = EmbeddingGenerationOptions::new();
        options
            .additional_properties
            .insert("task_type".into(), json!("NOT_A_TASK"));
        assert!(client()
            .build_body(&["q".to_string()], Some(&options))
            .is_err());
        // A non-string task names what was found rather than falling through
        // to "no task".
        let mut options = EmbeddingGenerationOptions::new();
        options
            .additional_properties
            .insert("task_type".into(), json!(7));
        let err = client()
            .build_body(&["q".to_string()], Some(&options))
            .expect_err("a number is not a task");
        assert!(err.to_string().contains("task name string"), "{err}");
    }

    #[test]
    fn a_title_outside_document_indexing_is_refused() {
        // Upstream refuses it too. Dropping it silently would leave the
        // caller believing the title was part of the embedding.
        let options = EmbeddingGenerationOptions::new()
            .with_task(GeminiEmbeddingTask::RetrievalQuery)
            .with_title("Ownership");
        let err = client()
            .build_body(&["q".to_string()], Some(&options))
            .expect_err("a title needs RETRIEVAL_DOCUMENT");
        assert!(
            err.to_string().contains("only applies when indexing"),
            "{err}"
        );
    }

    // endregion

    // region: request shape

    #[test]
    fn the_body_carries_one_request_per_value_with_the_resource_model() {
        let (body, model) = client()
            .build_body(&["a".to_string(), "b".to_string()], Some(&doc_options()))
            .unwrap();
        assert_eq!(model, DEFAULT_EMBEDDING_MODEL);
        let requests = body["requests"].as_array().unwrap();
        assert_eq!(requests.len(), 2);
        // `batchEmbedContents` wants the model on each entry, in resource form.
        assert_eq!(requests[0]["model"], json!("models/gemini-embedding-2"));
        assert_eq!(text_of(&body, 1), "title: none | text: b");
        // No `taskType` field: Embedding 2 takes the task from the text, and
        // a request field would be ignored. Sending one would suggest the
        // prefix is belt-and-braces when it is the only channel.
        assert!(requests[0].get("taskType").is_none(), "{body}");
    }

    #[test]
    fn dimensions_are_sent_per_request_and_zero_is_refused() {
        let options = doc_options();
        let (body, _) = client()
            .build_body(
                &["a".to_string()],
                Some(&EmbeddingGenerationOptions {
                    dimensions: Some(768),
                    ..options.clone()
                }),
            )
            .unwrap();
        assert_eq!(body["requests"][0]["outputDimensionality"], json!(768));

        // Omitted when unset, so the model's native width is used.
        let (body, _) = client()
            .build_body(&["a".to_string()], Some(&options))
            .unwrap();
        assert!(body["requests"][0].get("outputDimensionality").is_none());

        // Zero type-checks but the service rejects it; named here instead.
        let err = client()
            .build_body(
                &["a".to_string()],
                Some(&EmbeddingGenerationOptions {
                    dimensions: Some(0),
                    ..options
                }),
            )
            .expect_err("zero dimensions");
        assert!(err.to_string().contains("positive integer"), "{err}");
    }

    #[test]
    fn only_embedding_2_models_are_accepted() {
        // The prefix convention *is* Embedding 2's. Applying it to an older
        // model would embed the prefix as literal text — a silent accuracy
        // loss, which is why this is an allowlist and not a warning.
        for model in SUPPORTED_EMBEDDING_MODELS {
            assert!(client().try_with_model(model).is_ok(), "{model}");
        }
        for model in ["gemini-embedding-001", "text-embedding-004", ""] {
            assert!(client().try_with_model(model).is_err(), "{model}");
        }
        // Also when overridden per call, which bypasses the constructor.
        let options = EmbeddingGenerationOptions {
            model: Some("gemini-embedding-001".into()),
            ..doc_options()
        };
        assert!(client()
            .build_body(&["a".to_string()], Some(&options))
            .is_err());
        // A supported per-call override is honoured, and reported back so the
        // response is attributed to the model that produced it.
        let options = EmbeddingGenerationOptions {
            model: Some("gemini-embedding-2-preview".into()),
            ..doc_options()
        };
        let (body, model) = client()
            .build_body(&["a".to_string()], Some(&options))
            .unwrap();
        assert_eq!(model, "gemini-embedding-2-preview");
        assert_eq!(
            body["requests"][0]["model"],
            json!("models/gemini-embedding-2-preview")
        );
    }

    // endregion

    // region: response parsing

    #[test]
    fn vectors_are_parsed_in_order_and_attributed_to_the_model() {
        let value = json!({ "embeddings": [
            { "values": [0.1, 0.2] },
            { "values": [0.3, 0.4] },
        ]});
        let out = parse_embeddings_response(&value, 2, "gemini-embedding-2").unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].vector, vec![0.1f32, 0.2]);
        assert_eq!(out[1].vector, vec![0.3f32, 0.4]);
        assert_eq!(out[0].model.as_deref(), Some("gemini-embedding-2"));
    }

    #[test]
    fn a_short_or_malformed_response_errors_rather_than_misaligning() {
        // The caller pairs vectors with its own inputs by position, so a
        // silently shorter list would attach every later vector to the wrong
        // input — worse than a failed call, and invisible.
        let value = json!({ "embeddings": [{ "values": [0.1] }] });
        let err = parse_embeddings_response(&value, 2, "m").expect_err("count mismatch");
        assert!(err.to_string().contains("1 vectors for 2 inputs"), "{err}");

        assert!(parse_embeddings_response(&json!({}), 1, "m").is_err());
        assert!(parse_embeddings_response(&json!({ "embeddings": [{}] }), 1, "m").is_err());
        assert!(
            parse_embeddings_response(&json!({ "embeddings": [{ "values": ["x"] }] }), 1, "m")
                .is_err()
        );
    }

    // endregion

    // region: from_env

    #[test]
    fn from_env_reads_either_key_variable_and_the_model_override() {
        let c = GeminiEmbeddingClient::from_env_vars(|k| match k {
            "GEMINI_API_KEY" => Some("k1".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(c.model_name(), DEFAULT_EMBEDDING_MODEL);

        // GOOGLE_API_KEY is the fallback, as on the chat client.
        assert!(GeminiEmbeddingClient::from_env_vars(
            |k| (k == "GOOGLE_API_KEY").then(|| "k2".to_string())
        )
        .is_ok());

        // `GOOGLE_EMBEDDING_MODEL` is upstream's own variable name.
        let c = GeminiEmbeddingClient::from_env_vars(|k| match k {
            "GEMINI_API_KEY" => Some("k".into()),
            "GOOGLE_EMBEDDING_MODEL" => Some("gemini-embedding-2-preview".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(c.model_name(), "gemini-embedding-2-preview");

        // An unsupported override fails at construction rather than at the
        // first call.
        assert!(GeminiEmbeddingClient::from_env_vars(|k| match k {
            "GEMINI_API_KEY" => Some("k".into()),
            "GOOGLE_EMBEDDING_MODEL" => Some("text-embedding-004".into()),
            _ => None,
        })
        .is_err());

        assert!(GeminiEmbeddingClient::from_env_vars(|_| None).is_err());
    }

    // endregion
}
