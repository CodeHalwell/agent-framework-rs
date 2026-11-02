//! Azure Content Understanding: a [`ContextProvider`] that analyzes file
//! attachments and gives the model their content instead of the raw bytes.
//!
//! Port of upstream `agent_framework_azure_contentunderstanding`
//! (`ContentUnderstandingContextProvider`), behind this crate's
//! `content-understanding` feature, over the Content Understanding REST API.
//!
//! On each run the provider:
//!
//! 1. resumes analyses that outlasted `max_wait` on an earlier turn (their
//!    operation URL is kept in session state, so the state stays
//!    serializable);
//! 2. finds supported attachments ([`Content::Data`] and [`Content::Uri`]
//!    with a supported media type, sniffing magic bytes when the type is
//!    missing or `application/octet-stream`), **removes them from the run's
//!    input**, and analyzes each with the provider's analyzer or the
//!    prebuilt one for its media type (`prebuilt-documentSearch`,
//!    `prebuilt-audioSearch`, `prebuilt-videoSearch`);
//! 3. injects each finished document as YAML front matter (source and
//!    extracted fields) followed by its markdown, plus a note telling the
//!    model about it; an analysis still running after `max_wait` is reported
//!    as in progress and resumed on the next turn;
//! 4. registers `list_documents`, which reports every document of the
//!    session with its status.
//!
//! With a [`FileSearchConfig`], finished documents are uploaded to a vector
//! store through its [`FileSearchBackend`] instead of being injected, and
//! the given `file_search` tool is offered.
//!
//! # Divergences
//!
//! - **Filenames.** Upstream reads a per-attachment `filename`,
//!   `analyzer_id` and `content_range` from the content's
//!   `additional_properties`, which Rust content items do not have. The
//!   document key here is a `name=` or `filename=` parameter of a data URI
//!   (`data:application/pdf;name=invoice.pdf;base64,...`), else the URL's
//!   last path segment, else a generated id. The analyzer and content range
//!   are provider-wide ([`ContentUnderstandingContextProvider::with_analyzer`],
//!   [`ContentUnderstandingContextProvider::with_content_range`]).
//! - **Rendering.** Upstream renders with the Azure SDK's `to_llm_input`;
//!   this renders the same parts (front matter with `source` and the
//!   fields' values, then the markdown) in its own words.
//! - **File search backends.** [`FileSearchBackend`] is a trait; no built-in
//!   OpenAI or Foundry backend ships yet.

use std::sync::Arc;
use std::time::{Duration, Instant};

use agent_framework_core::error::{Error, Result};
use agent_framework_core::memory::{ContextProvider, SessionContext};
use agent_framework_core::session::AgentSession;
use agent_framework_core::tools::{FunctionTool, ToolDefinition};
use agent_framework_core::types::{Content, Message};
use async_trait::async_trait;
use base64::Engine as _;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::TokenCredential;

/// The default `source_id`.
pub const DEFAULT_CONTENT_UNDERSTANDING_SOURCE_ID: &str = "azure_contentunderstanding";
/// The REST API version.
pub const CONTENT_UNDERSTANDING_API_VERSION: &str = "2025-11-01";
/// The Entra ID scope for Content Understanding.
pub const COGNITIVE_SERVICES_SCOPE: &str = "https://cognitiveservices.azure.com/.default";
/// The analyzer for documents and images.
pub const DEFAULT_ANALYZER: &str = "prebuilt-documentSearch";

/// The media types Content Understanding analyzes.
pub const SUPPORTED_MEDIA_TYPES: &[&str] = &[
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/bmp",
    "image/heif",
    "image/heic",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "text/plain",
    "text/html",
    "text/markdown",
    "text/rtf",
    "text/xml",
    "application/xml",
    "message/rfc822",
    "application/vnd.ms-outlook",
    "audio/wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/m4a",
    "audio/flac",
    "audio/ogg",
    "audio/opus",
    "audio/webm",
    "audio/x-ms-wma",
    "audio/aac",
    "audio/amr",
    "audio/3gpp",
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/webm",
    "video/x-flv",
    "video/x-ms-wmv",
    "video/x-ms-asf",
    "video/x-matroska",
];

/// Where a vector store upload goes, for [`FileSearchConfig`].
#[async_trait]
pub trait FileSearchBackend: Send + Sync {
    /// Upload `content` as `filename` into `vector_store_id` and wait for it
    /// to be indexed, returning the file id.
    async fn upload(&self, vector_store_id: &str, filename: &str, content: &str) -> Result<String>;
}

/// Upload analyzed documents to a vector store instead of injecting them.
#[derive(Clone)]
pub struct FileSearchConfig {
    /// The upload backend.
    pub backend: Arc<dyn FileSearchBackend>,
    /// The vector store to upload into.
    pub vector_store_id: String,
    /// The `file_search` tool offered to the model.
    pub file_search_tool: ToolDefinition,
}

/// The lifecycle of a tracked document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DocumentStatus {
    /// The analysis is running.
    Analyzing,
    /// Analyzed; the vector store upload is running.
    Uploading,
    /// Ready.
    Ready,
    /// Analysis or upload failed.
    Failed,
}

/// A tracked document, kept in session state.
#[derive(Debug, Clone, PartialEq, serde::Serialize, Deserialize)]
pub struct DocumentEntry {
    /// Its status.
    pub status: DocumentStatus,
    /// Its key (the filename).
    pub filename: String,
    /// Its media type.
    pub media_type: String,
    /// The analyzer used.
    pub analyzer_id: String,
    /// When analysis finished (Unix seconds).
    pub analyzed_at: Option<u64>,
    /// How long analysis took.
    pub analysis_duration_s: Option<f64>,
    /// How long the upload took.
    pub upload_duration_s: Option<f64>,
    /// The rendered text.
    pub result: Option<String>,
    /// The failure.
    pub error: Option<String>,
}

#[derive(Debug, Clone, Default, serde::Serialize, Deserialize)]
struct State {
    #[serde(default)]
    documents: std::collections::BTreeMap<String, DocumentEntry>,
    /// doc key → (operation URL, analyzer id).
    #[serde(default, rename = "_pending_tokens")]
    pending: std::collections::BTreeMap<String, Pending>,
}

#[derive(Debug, Clone, serde::Serialize, Deserialize)]
struct Pending {
    continuation_token: String,
    analyzer_id: String,
}

enum Auth {
    ApiKey(String),
    Credential(Arc<dyn TokenCredential>),
}

/// The provider. See the [module docs](self).
#[derive(Clone)]
pub struct ContentUnderstandingContextProvider {
    inner: Arc<Inner>,
}

struct Inner {
    http: reqwest::Client,
    endpoint: String,
    auth: Auth,
    source_id: String,
    analyzer_id: Option<String>,
    content_range: Option<String>,
    max_wait: Option<Duration>,
    include_markdown: bool,
    include_fields: bool,
    file_search: Option<FileSearchConfig>,
    poll_interval: Duration,
}

impl std::fmt::Debug for ContentUnderstandingContextProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContentUnderstandingContextProvider")
            .field("endpoint", &self.inner.endpoint)
            .field("source_id", &self.inner.source_id)
            .field("analyzer_id", &self.inner.analyzer_id)
            .finish_non_exhaustive()
    }
}

impl ContentUnderstandingContextProvider {
    fn build(endpoint: &str, auth: Auth) -> Self {
        Self {
            inner: Arc::new(Inner {
                http: crate::azure_http_client(env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION")),
                endpoint: endpoint.trim_end_matches('/').to_string(),
                auth,
                source_id: DEFAULT_CONTENT_UNDERSTANDING_SOURCE_ID.into(),
                analyzer_id: None,
                content_range: None,
                max_wait: Some(Duration::from_secs(5)),
                include_markdown: true,
                include_fields: true,
                file_search: None,
                poll_interval: Duration::from_millis(500),
            }),
        }
    }

    fn update(self, f: impl FnOnce(&mut Inner)) -> Self {
        let mut inner = Arc::try_unwrap(self.inner).unwrap_or_else(|arc| Inner {
            http: arc.http.clone(),
            endpoint: arc.endpoint.clone(),
            auth: match &arc.auth {
                Auth::ApiKey(k) => Auth::ApiKey(k.clone()),
                Auth::Credential(c) => Auth::Credential(c.clone()),
            },
            source_id: arc.source_id.clone(),
            analyzer_id: arc.analyzer_id.clone(),
            content_range: arc.content_range.clone(),
            max_wait: arc.max_wait,
            include_markdown: arc.include_markdown,
            include_fields: arc.include_fields,
            file_search: arc.file_search.clone(),
            poll_interval: arc.poll_interval,
        });
        f(&mut inner);
        Self {
            inner: Arc::new(inner),
        }
    }

    /// A provider for the Foundry / AI Services `endpoint` with a key.
    pub fn new(endpoint: &str, api_key: impl Into<String>) -> Self {
        Self::build(endpoint, Auth::ApiKey(api_key.into()))
    }

    /// As [`new`](Self::new), with Entra ID (scoped to
    /// [`COGNITIVE_SERVICES_SCOPE`]).
    pub fn with_token_credential(endpoint: &str, credential: Arc<dyn TokenCredential>) -> Self {
        Self::build(endpoint, Auth::Credential(credential))
    }

    /// From `AZURE_CONTENTUNDERSTANDING_ENDPOINT` and, when set,
    /// `AZURE_CONTENTUNDERSTANDING_API_KEY` (else a `DefaultAzureCredential`).
    pub fn from_env() -> Result<Self> {
        let endpoint = std::env::var("AZURE_CONTENTUNDERSTANDING_ENDPOINT").map_err(|_| {
            Error::Configuration("AZURE_CONTENTUNDERSTANDING_ENDPOINT is not set.".into())
        })?;
        Ok(match std::env::var("AZURE_CONTENTUNDERSTANDING_API_KEY") {
            Ok(key) if !key.is_empty() => Self::new(&endpoint, key),
            _ => Self::with_token_credential(
                &endpoint,
                Arc::new(crate::DefaultAzureCredential::new(COGNITIVE_SERVICES_SCOPE)),
            ),
        })
    }

    /// The `source_id` (default `azure_contentunderstanding`).
    pub fn with_source_id(self, source_id: impl Into<String>) -> Self {
        let source_id = source_id.into();
        self.update(|i| i.source_id = source_id)
    }
    /// Analyze every file with this analyzer (default: by media type).
    pub fn with_analyzer(self, analyzer_id: impl Into<String>) -> Self {
        let id = analyzer_id.into();
        self.update(|i| i.analyzer_id = Some(id))
    }
    /// Analyze only this range (pages such as `"1-3"`, or milliseconds for
    /// audio and video).
    pub fn with_content_range(self, range: impl Into<String>) -> Self {
        let range = range.into();
        self.update(|i| i.content_range = Some(range))
    }
    /// How long to wait for an analysis before reporting it as in progress
    /// (default 5 s; `None` waits until it finishes).
    pub fn with_max_wait(self, max_wait: Option<Duration>) -> Self {
        self.update(|i| i.max_wait = max_wait)
    }
    /// Which sections to give the model (default both).
    pub fn with_output_sections(self, markdown: bool, fields: bool) -> Self {
        self.update(|i| {
            i.include_markdown = markdown;
            i.include_fields = fields;
        })
    }
    /// Upload documents to a vector store instead of injecting them.
    pub fn with_file_search(self, config: FileSearchConfig) -> Self {
        self.update(|i| i.file_search = Some(config))
    }
    /// How often to poll a running analysis (default 500 ms).
    pub fn with_poll_interval(self, interval: Duration) -> Self {
        self.update(|i| i.poll_interval = interval)
    }

    /// The analyzer for a media type when none is configured.
    pub fn analyzer_for(&self, media_type: &str) -> String {
        if let Some(id) = &self.inner.analyzer_id {
            return id.clone();
        }
        if media_type.starts_with("audio/") {
            "prebuilt-audioSearch".into()
        } else if media_type.starts_with("video/") {
            "prebuilt-videoSearch".into()
        } else {
            DEFAULT_ANALYZER.into()
        }
    }

    async fn authorize(&self, request: reqwest::RequestBuilder) -> Result<reqwest::RequestBuilder> {
        Ok(match &self.inner.auth {
            Auth::ApiKey(key) => request.header("Ocp-Apim-Subscription-Key", key),
            Auth::Credential(c) => {
                request.bearer_auth(c.get_token_for_scope(COGNITIVE_SERVICES_SCOPE).await?)
            }
        })
    }

    /// Start an analysis, returning its operation URL.
    async fn begin(
        &self,
        analyzer: &str,
        url: Option<&str>,
        binary: Option<(&[u8], &str)>,
    ) -> Result<String> {
        let base = format!(
            "{}/contentunderstanding/analyzers/{analyzer}",
            self.inner.endpoint
        );
        let request = match (url, binary) {
            (Some(url), _) => {
                let mut input = json!({"url": url});
                if let Some(range) = &self.inner.content_range {
                    input["contentRange"] = json!(range);
                }
                self.inner
                    .http
                    .post(format!(
                        "{base}:analyze?api-version={CONTENT_UNDERSTANDING_API_VERSION}"
                    ))
                    .json(&json!({"inputs": [input]}))
            }
            (None, Some((bytes, media_type))) => {
                let mut url =
                    format!("{base}:analyzeBinary?api-version={CONTENT_UNDERSTANDING_API_VERSION}");
                if let Some(range) = &self.inner.content_range {
                    url.push_str(&format!("&range={range}"));
                }
                self.inner
                    .http
                    .post(url)
                    .header("Content-Type", media_type)
                    .body(bytes.to_vec())
            }
            (None, None) => return Err(Error::Configuration("no file data".into())),
        };
        let response = self.authorize(request).await?.send().await.map_err(|e| {
            Error::Service(format!(
                "Content Understanding request failed: {}",
                agent_framework_core::error::error_chain(&e)
            ))
        })?;
        let status = response.status();
        let location = response
            .headers()
            .get("Operation-Location")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(Error::Service(format!(
                "Content Understanding returned {status}: {text}"
            )));
        }
        location.ok_or_else(|| {
            Error::Service("Content Understanding returned no Operation-Location.".into())
        })
    }

    /// Poll an operation until it finishes or `wait` passes: `Ok(Some)` with
    /// the result, `Ok(None)` while still running.
    async fn poll(&self, operation: &str, wait: Option<Duration>) -> Result<Option<Value>> {
        let deadline = wait.map(|w| Instant::now() + w);
        loop {
            let request = self.authorize(self.inner.http.get(operation)).await?;
            let response = request.send().await.map_err(|e| {
                Error::Service(format!(
                    "Content Understanding poll failed: {}",
                    agent_framework_core::error::error_chain(&e)
                ))
            })?;
            let status = response.status();
            let body: Value = response.json().await.unwrap_or(Value::Null);
            if !status.is_success() {
                return Err(Error::Service(format!(
                    "Content Understanding returned {status}: {body}"
                )));
            }
            match body
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_ascii_lowercase()
                .as_str()
            {
                "succeeded" => return Ok(Some(body.get("result").cloned().unwrap_or(Value::Null))),
                "failed" | "canceled" | "cancelled" => {
                    let detail = body
                        .pointer("/error/message")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                        .unwrap_or_else(|| body.to_string());
                    return Err(Error::Service(format!("analysis failed: {detail}")));
                }
                _ => {}
            }
            if let Some(deadline) = deadline {
                let now = Instant::now();
                if now >= deadline {
                    return Ok(None);
                }
                tokio::time::sleep(self.inner.poll_interval.min(deadline - now)).await;
            } else {
                tokio::time::sleep(self.inner.poll_interval).await;
            }
        }
    }

    /// The LLM text for an analysis result.
    pub fn render(&self, result: &Value, filename: &str) -> String {
        render_for_llm(
            result,
            filename,
            self.inner.include_markdown,
            self.inner.include_fields,
        )
    }

    fn load_state(&self, session: &AgentSession) -> State {
        session
            .state
            .get(&self.inner.source_id)
            .and_then(|v| serde_json::from_value(v).ok())
            .unwrap_or_default()
    }

    fn save_state(&self, session: &AgentSession, state: &State) {
        if let Ok(v) = serde_json::to_value(state) {
            session.state.insert(&self.inner.source_id, v);
        }
    }

    fn list_documents_tool(&self, session: &AgentSession) -> ToolDefinition {
        let (provider, session) = (self.clone(), session.clone());
        FunctionTool::new(
            "list_documents",
            "List all documents that have been uploaded in this session with their analysis status \
             (analyzing, uploading, ready, or failed).",
            json!({"type": "object", "properties": {}}),
            move |_| {
                let (provider, session) = (provider.clone(), session.clone());
                async move {
                    let state = provider.load_state(&session);
                    let entries: Vec<Value> = state
                        .documents
                        .iter()
                        .map(|(name, e)| {
                            json!({
                                "name": name,
                                "status": e.status,
                                "media_type": e.media_type,
                                "analyzed_at": e.analyzed_at,
                                "analysis_duration_s": e.analysis_duration_s,
                                "upload_duration_s": e.upload_duration_s,
                            })
                        })
                        .collect();
                    Ok(Value::String(serde_json::to_string_pretty(&entries)?))
                }
            },
        )
        .into_definition()
    }

    /// Upload a ready document; returns whether it was indexed.
    async fn upload(&self, entry: &mut DocumentEntry, timeout: Option<Duration>) -> bool {
        let Some(fs) = &self.inner.file_search else {
            return false;
        };
        let Some(text) = entry.result.clone().filter(|t| has_renderable_body(t)) else {
            return false;
        };
        entry.status = DocumentStatus::Uploading;
        let started = Instant::now();
        let filename = format!("{}.md", entry.filename);
        let upload = fs.backend.upload(&fs.vector_store_id, &filename, &text);
        let outcome = match timeout {
            Some(t) => tokio::time::timeout(t, upload).await.ok(),
            None => Some(upload.await),
        };
        match outcome {
            Some(Ok(_)) => {
                entry.status = DocumentStatus::Ready;
                entry.upload_duration_s = Some(round2(started.elapsed()));
                true
            }
            Some(Err(e)) => {
                entry.status = DocumentStatus::Failed;
                entry.error = Some(e.to_string());
                false
            }
            None => false, // Still indexing; reported as such.
        }
    }
}

fn round2(d: Duration) -> f64 {
    (d.as_secs_f64() * 100.0).round() / 100.0
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Whether `text` has content past its YAML front matter.
pub fn has_renderable_body(text: &str) -> bool {
    let trimmed = text.trim_start();
    let Some(rest) = trimmed.strip_prefix("---") else {
        return !text.trim().is_empty();
    };
    match rest.find("\n---") {
        Some(end) => !rest[end + 4..].trim().is_empty(),
        None => !text.trim().is_empty(),
    }
}

/// A field's value, from the first `value*` property (recursing into
/// arrays and objects).
fn field_value(field: &Value) -> Value {
    let Some(map) = field.as_object() else {
        return field.clone();
    };
    for (key, value) in map {
        match key.as_str() {
            "valueArray" => {
                return Value::Array(
                    value
                        .as_array()
                        .into_iter()
                        .flatten()
                        .map(field_value)
                        .collect(),
                );
            }
            "valueObject" => {
                let obj: Map<String, Value> = value
                    .as_object()
                    .into_iter()
                    .flatten()
                    .map(|(k, v)| (k.clone(), field_value(v)))
                    .collect();
                return Value::Object(obj);
            }
            k if k.starts_with("value") => return value.clone(),
            _ => {}
        }
    }
    Value::Null
}

fn yaml_scalar(value: &Value) -> String {
    match value {
        Value::String(s) => serde_json::to_string(s).unwrap_or_default(),
        Value::Null => "null".into(),
        other => other.to_string(),
    }
}

/// Render an analysis result: YAML front matter (`source`, then `fields`
/// when asked for) followed by each content's markdown.
pub fn render_for_llm(result: &Value, filename: &str, markdown: bool, fields: bool) -> String {
    let contents = result
        .get("contents")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut front = vec![
        "---".to_string(),
        format!("source: {}", yaml_scalar(&json!(filename))),
    ];
    if fields {
        let mut lines = Vec::new();
        for content in &contents {
            for (name, field) in content
                .get("fields")
                .and_then(Value::as_object)
                .into_iter()
                .flatten()
            {
                lines.push(format!(
                    "  {}: {}",
                    yaml_scalar(&json!(name)),
                    yaml_scalar(&field_value(field))
                ));
            }
        }
        if !lines.is_empty() {
            front.push("fields:".into());
            front.extend(lines);
        }
    }
    front.push("---".into());
    let mut out = front.join("\n");
    if markdown {
        let body: Vec<&str> = contents
            .iter()
            .filter_map(|c| c.get("markdown").and_then(Value::as_str))
            .filter(|m| !m.trim().is_empty())
            .collect();
        if !body.is_empty() {
            out.push('\n');
            out.push_str(&body.join("\n\n"));
        }
    }
    out
}

/// Sniff a media type from magic bytes.
pub fn sniff_media_type(bytes: &[u8]) -> Option<&'static str> {
    let starts = |sig: &[u8]| bytes.starts_with(sig);
    Some(if starts(b"%PDF") {
        "application/pdf"
    } else if starts(&[0xFF, 0xD8, 0xFF]) {
        "image/jpeg"
    } else if starts(&[0x89, b'P', b'N', b'G']) {
        "image/png"
    } else if starts(b"II*\0") || starts(b"MM\0*") {
        "image/tiff"
    } else if starts(b"BM") {
        "image/bmp"
    } else if bytes.len() > 11 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
        "audio/wav"
    } else if bytes.len() > 11 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"AVI " {
        "video/x-msvideo"
    } else if starts(b"ID3") || starts(&[0xFF, 0xFB]) {
        "audio/mpeg"
    } else if starts(b"fLaC") {
        "audio/flac"
    } else if starts(b"OggS") {
        "audio/ogg"
    } else if bytes.len() > 11 && &bytes[4..8] == b"ftyp" {
        let brand = &bytes[8..12];
        if brand == b"M4A " {
            "audio/m4a"
        } else if brand == b"qt  " {
            "video/quicktime"
        } else if brand.starts_with(b"hei") || brand.starts_with(b"mif1") {
            "image/heic"
        } else {
            "video/mp4"
        }
    } else if starts(&[0x1A, 0x45, 0xDF, 0xA3]) {
        "video/x-matroska"
    } else {
        return None;
    })
}

fn media_type_from_name(name: &str) -> Option<&'static str> {
    let ext = name.rsplit('.').next()?.to_ascii_lowercase();
    Some(match ext.as_str() {
        "pdf" => "application/pdf",
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "tif" | "tiff" => "image/tiff",
        "bmp" => "image/bmp",
        "heic" => "image/heic",
        "docx" => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xlsx" => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "pptx" => "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "txt" => "text/plain",
        "html" | "htm" => "text/html",
        "md" => "text/markdown",
        "xml" => "application/xml",
        "eml" => "message/rfc822",
        "wav" => "audio/wav",
        "mp3" => "audio/mpeg",
        "m4a" => "audio/m4a",
        "flac" => "audio/flac",
        "mp4" => "video/mp4",
        "mov" => "video/quicktime",
        "webm" => "video/webm",
        "mkv" => "video/x-matroska",
        _ => return None,
    })
}

/// Remove control characters, collapse whitespace and cap at 255
/// characters, so a crafted filename cannot inject instructions.
pub fn sanitize_doc_key(raw: &str) -> String {
    let cleaned: String = raw.chars().filter(|c| !c.is_control()).collect();
    let cleaned = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
    if cleaned.is_empty() {
        format!("doc_{}", &uuid::Uuid::new_v4().simple().to_string()[..8])
    } else {
        cleaned.chars().take(255).collect()
    }
}

/// A file found in the input.
struct Attachment {
    key: String,
    media_type: String,
    url: Option<String>,
    bytes: Option<Vec<u8>>,
}

fn url_basename(uri: &str) -> Option<String> {
    let path = uri.split(['?', '#']).next()?;
    let base = path.trim_end_matches('/').rsplit('/').next()?;
    (!base.is_empty()).then(|| base.to_string())
}

/// A data URI's `key=value` parameters and decoded payload.
type ParsedDataUri = (Vec<(String, String)>, Vec<u8>);

/// `(params, base64 payload)` of a data URI.
fn parse_data_uri(uri: &str) -> Option<ParsedDataUri> {
    let rest = uri.strip_prefix("data:")?;
    let (meta, data) = rest.split_once(',')?;
    let params = meta
        .split(';')
        .skip(1)
        .filter_map(|p| {
            p.split_once('=')
                .map(|(k, v)| (k.to_ascii_lowercase(), v.to_string()))
        })
        .collect();
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data.trim())
        .ok()?;
    Some((params, bytes))
}

/// Take the supported attachments out of `messages`.
fn detect_and_strip(messages: &mut [Message]) -> Vec<Attachment> {
    let supported = |t: &str| SUPPORTED_MEDIA_TYPES.contains(&t);
    let mut found = Vec::new();
    for message in messages.iter_mut() {
        let mut kept = Vec::with_capacity(message.contents.len());
        for content in std::mem::take(&mut message.contents) {
            let attachment = match &content {
                Content::Data(d) => parse_data_uri(&d.uri).and_then(|(params, bytes)| {
                    let name = params
                        .iter()
                        .find(|(k, _)| k == "name" || k == "filename")
                        .map(|(_, v)| v.clone());
                    let declared = d
                        .media_type
                        .clone()
                        .filter(|t| !t.is_empty() && t != "application/octet-stream");
                    let media_type = match declared {
                        Some(t) => t,
                        None => sniff_media_type(&bytes)
                            .or_else(|| name.as_deref().and_then(media_type_from_name))?
                            .to_string(),
                    };
                    supported(&media_type).then(|| Attachment {
                        key: name
                            .map(|n| sanitize_doc_key(&n))
                            .unwrap_or_else(|| sanitize_doc_key("")),
                        media_type,
                        url: None,
                        bytes: Some(bytes),
                    })
                }),
                Content::Uri(u) if !u.uri.starts_with("data:") => {
                    let declared = Some(u.media_type.clone())
                        .filter(|t| !t.is_empty() && t != "application/octet-stream");
                    let base = url_basename(&u.uri);
                    let media_type = declared.or_else(|| {
                        base.as_deref()
                            .and_then(media_type_from_name)
                            .map(str::to_string)
                    });
                    media_type
                        .filter(|t| supported(t))
                        .map(|media_type| Attachment {
                            key: base
                                .map(|b| sanitize_doc_key(&b))
                                .unwrap_or_else(|| sanitize_doc_key("")),
                            media_type,
                            url: Some(u.uri.clone()),
                            bytes: None,
                        })
                }
                _ => None,
            };
            match attachment {
                Some(a) => found.push(a),
                None => kept.push(content),
            }
        }
        message.contents = kept;
    }
    found
}

#[async_trait]
impl ContextProvider for ContentUnderstandingContextProvider {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        let session = ctx.session.clone().ok_or_else(|| {
            Error::Configuration("ContentUnderstandingContextProvider requires a session.".into())
        })?;
        let mut state = self.load_state(&session);
        let file_search = self.inner.file_search.is_some();
        let note = |ctx: &mut SessionContext, text: String| ctx.messages.push(Message::user(text));

        // 1. Resume analyses left running on earlier turns.
        let wait = Some(
            self.inner
                .max_wait
                .unwrap_or(Duration::from_secs(10))
                .max(Duration::from_secs(10)),
        );
        let pending: Vec<(String, Pending)> =
            std::mem::take(&mut state.pending).into_iter().collect();
        for (key, token) in pending {
            let Some(mut entry) = state.documents.get(&key).cloned() else {
                continue;
            };
            match self.poll(&token.continuation_token, wait).await {
                Ok(None) => {
                    state.pending.insert(key, token);
                    continue;
                }
                Ok(Some(result)) => {
                    let rendered = self.render(&result, &entry.filename);
                    entry.status = DocumentStatus::Ready;
                    entry.analyzed_at = Some(now_secs());
                    entry.result = Some(rendered.clone());
                    entry.error = None;
                    if file_search {
                        self.upload(&mut entry, self.inner.max_wait).await;
                        note(ctx, format!(
                            "Document '{}' analysis is now complete. The document is being indexed in the vector \
                             store and will become searchable via file_search shortly.",
                            entry.filename
                        ));
                    } else {
                        ctx.messages.push(Message::user(rendered));
                        note(ctx, format!(
                            "Document '{}' analysis is now complete. The content is provided above.",
                            entry.filename
                        ));
                    }
                }
                Err(e) => {
                    entry.status = DocumentStatus::Failed;
                    entry.analyzed_at = Some(now_secs());
                    entry.error = Some(e.to_string());
                    note(
                        ctx,
                        format!("Document '{}' analysis failed: {e}", entry.filename),
                    );
                }
            }
            state.documents.insert(key, entry);
        }

        // 2. New attachments, taken out of the input.
        for attachment in detect_and_strip(&mut ctx.input_messages) {
            let key = attachment.key.clone();
            if state.documents.contains_key(&key) {
                note(ctx, format!(
                    "The user tried to upload '{key}', but a file with that name was already uploaded earlier in \
                     this session. The new upload was rejected and was not analyzed. Tell the user that a file \
                     with the same name already exists and they need to rename the file before uploading again."
                ));
                continue;
            }
            let analyzer = self.analyzer_for(&attachment.media_type);
            let started = Instant::now();
            let mut entry = DocumentEntry {
                status: DocumentStatus::Analyzing,
                filename: key.clone(),
                media_type: attachment.media_type.clone(),
                analyzer_id: analyzer.clone(),
                analyzed_at: None,
                analysis_duration_s: None,
                upload_duration_s: None,
                result: None,
                error: None,
            };
            let outcome = async {
                let operation = self
                    .begin(
                        &analyzer,
                        attachment.url.as_deref(),
                        attachment
                            .bytes
                            .as_deref()
                            .map(|b| (b, attachment.media_type.as_str())),
                    )
                    .await?;
                let result = self.poll(&operation, self.inner.max_wait).await?;
                Ok::<_, Error>((operation, result))
            }
            .await;
            match outcome {
                Ok((operation, None)) => {
                    state.pending.insert(
                        key.clone(),
                        Pending {
                            continuation_token: operation,
                            analyzer_id: analyzer,
                        },
                    );
                    note(
                        ctx,
                        format!(
                            "Document '{key}' is being analyzed. Ask about it again in a moment."
                        ),
                    );
                }
                Ok((_, Some(result))) => {
                    let rendered = self.render(&result, &key);
                    entry.status = DocumentStatus::Ready;
                    entry.analyzed_at = Some(now_secs());
                    entry.analysis_duration_s = Some(round2(started.elapsed()));
                    entry.result = Some(rendered.clone());
                    if file_search {
                        let remaining = self
                            .inner
                            .max_wait
                            .map(|w| w.saturating_sub(started.elapsed()));
                        if self.upload(&mut entry, remaining).await {
                            note(ctx, format!(
                                "The user just uploaded '{key}'. It has been analyzed using Azure Content \
                                 Understanding and indexed in a vector store. When using file_search, include \
                                 '{key}' in your query to retrieve content from this specific document."
                            ));
                        } else if entry.error.is_some() {
                            note(ctx, format!(
                                "Document '{key}' was analyzed but failed to upload to the vector store. The \
                                 document content is not available for search."
                            ));
                        } else {
                            note(ctx, format!(
                                "Document '{key}' has been analyzed and is being indexed. Ask about it again in a \
                                 moment."
                            ));
                        }
                    } else {
                        ctx.messages.push(Message::user(rendered));
                        note(ctx, format!(
                            "The user just uploaded '{key}'. It has been analyzed using Azure Content \
                             Understanding. The document content (markdown) and extracted fields (YAML front \
                             matter) are provided above. If the user's question is ambiguous, prioritize this most \
                             recently uploaded document. Use specific field values and cite page numbers when \
                             answering."
                        ));
                    }
                }
                Err(e) => {
                    tracing::warn!(document = %key, error = %e, "Content Understanding analysis failed");
                    entry.status = DocumentStatus::Failed;
                    entry.analyzed_at = Some(now_secs());
                    entry.analysis_duration_s = Some(round2(started.elapsed()));
                    entry.error = Some(e.to_string());
                    note(ctx, format!("Could not analyze '{key}': {e}"));
                }
            }
            state.documents.insert(key, entry);
        }

        self.save_state(&session, &state);
        if !state.documents.is_empty() {
            ctx.tools.push(self.list_documents_tool(&session));
        }
        if let Some(fs) = &self.inner.file_search {
            ctx.tools.push(fs.file_search_tool.clone());
            ctx.add_instructions(
                "Tool usage guidelines:\n- Use file_search ONLY when answering questions about document content.\n\
                 - Use list_documents() for status queries (e.g. 'list docs', 'what's uploaded?').\n\
                 - Do NOT call file_search for status queries \u{2014} it wastes tokens.",
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::types::DataContent;

    #[test]
    fn renders_front_matter_and_markdown() {
        let result = json!({"contents": [{"markdown": "# Invoice\nTotal: 5", "fields": {
            "Total": {"type": "number", "valueNumber": 5},
            "Lines": {"type": "array", "valueArray": [{"type": "string", "valueString": "a"}]}
        }}]});
        let text = render_for_llm(&result, "inv.pdf", true, true);
        assert!(
            text.starts_with("---\nsource: \"inv.pdf\"\nfields:\n"),
            "{text}"
        );
        assert!(text.contains("  \"Total\": 5"), "{text}");
        assert!(text.contains("  \"Lines\": [\"a\"]"), "{text}");
        assert!(text.ends_with("---\n# Invoice\nTotal: 5"), "{text}");
        assert!(has_renderable_body(&text));
        assert!(!has_renderable_body(&render_for_llm(
            &json!({"contents": []}),
            "x",
            true,
            true
        )));
    }

    #[test]
    fn attachments_are_detected_named_and_stripped() {
        let pdf = DataContent::from_bytes(b"%PDF-1.7 ...", "application/octet-stream");
        let named = DataContent {
            uri: format!(
                "data:application/pdf;name=invoice\n.pdf;base64,{}",
                base64::engine::general_purpose::STANDARD.encode(b"%PDF")
            ),
            media_type: Some("application/pdf".into()),
        };
        let mut messages = vec![Message::with_contents(
            agent_framework_core::types::Role::user(),
            vec![
                Content::text("summarize these"),
                Content::Data(pdf),
                Content::Data(named),
                Content::Uri(agent_framework_core::types::UriContent {
                    uri: "https://x.test/files/report.docx?sig=1".into(),
                    media_type: String::new(),
                }),
                Content::Uri(agent_framework_core::types::UriContent {
                    uri: "https://x.test/cat.gif".into(),
                    media_type: "image/gif".into(),
                }),
            ],
        )];
        let found = detect_and_strip(&mut messages);
        assert_eq!(found.len(), 3);
        assert_eq!(found[0].media_type, "application/pdf", "sniffed");
        assert!(found[0].key.starts_with("doc_"));
        assert_eq!(found[1].key, "invoice.pdf", "control characters removed");
        assert_eq!(found[2].key, "report.docx");
        assert!(found[2].url.is_some());
        assert_eq!(
            messages[0].contents.len(),
            2,
            "text and the unsupported gif stay"
        );
    }
}
