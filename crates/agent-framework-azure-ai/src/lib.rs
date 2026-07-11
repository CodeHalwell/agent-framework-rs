//! # agent-framework-azure-ai
//!
//! An Azure AI Foundry (persistent **agents**) [`ChatClient`] for
//! `agent-framework-rs`. Talks the Azure AI Agents data-plane REST API directly
//! (no Azure SDK dependency): agents (assistants), threads, messages, and runs,
//! with streaming (SSE) and non-streaming (poll) execution, tool-call
//! round-tripping via `submit_tool_outputs`, and Microsoft Entra ID
//! authentication through a [`TokenCredential`].
//!
//! The service thread id travels as the response's
//! [`conversation_id`](agent_framework_core::types::ChatResponse::conversation_id):
//! a thread is created on the first turn and surfaced back so a follow-up turn
//! (set `ChatOptions::conversation_id`) continues the same conversation.
//!
//! ```no_run
//! use std::sync::Arc;
//! use agent_framework_azure::AzureCliCredential;
//! use agent_framework_azure_ai::AzureAIAgentClient;
//! use agent_framework_core::prelude::*;
//!
//! # async fn demo() -> Result<()> {
//! let credential = Arc::new(AzureCliCredential::new(
//!     agent_framework_azure_ai::AI_FOUNDRY_SCOPE,
//! ));
//! // Auto-creates (and, on `close`, deletes) a transient agent for `gpt-4o`.
//! let client = AzureAIAgentClient::new(
//!     "https://my-project.services.ai.azure.com",
//!     "gpt-4o",
//!     credential,
//! );
//! let agent = ChatAgent::builder(client).instructions("You are concise.").build();
//! let reply = agent.run_once("Say hi").await?;
//! println!("{}", reply.text());
//! # Ok(())
//! # }
//! ```
//!
//! Use an already-created (persistent) agent instead of an auto-created one:
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use agent_framework_azure::AzureCliCredential;
//! use agent_framework_azure_ai::{AzureAIAgentClient, AI_FOUNDRY_SCOPE};
//! # fn demo() {
//! let credential = Arc::new(AzureCliCredential::new(AI_FOUNDRY_SCOPE));
//! let client = AzureAIAgentClient::with_existing_agent(
//!     "https://my-project.services.ai.azure.com",
//!     "asst_abc123",
//!     credential,
//! );
//! # let _ = client;
//! # }
//! ```

mod convert;
mod sse;

pub use convert::AI_FOUNDRY_SCOPE;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_framework_azure::TokenCredential;
use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::error::{Error, Result};
use agent_framework_core::types::{ChatMessage, ChatOptions, ChatResponse, Role};
use futures::StreamExt;
use serde_json::{json, Map, Value};

use convert::PreparedMessages;

/// The default Azure AI Agents data-plane API version sent as `?api-version=`.
///
/// The Azure AI Agents service is not vendored locally, so this is the
/// documented GA value; override with [`AzureAIAgentClient::with_api_version`]
/// if your project pins a different one.
pub const DEFAULT_API_VERSION: &str = "2025-05-01";

/// How often a non-streaming run is polled for completion.
const POLL_INTERVAL: Duration = Duration::from_millis(500);
/// Upper bound on poll iterations (~5 minutes) before giving up.
const MAX_POLLS: usize = 600;

/// Parse a `Retry-After` header into a delay in seconds (see the OpenAI/Azure
/// OpenAI clients), so a [`RetryingChatClient`](agent_framework_core::client::RetryingChatClient)
/// honors the server's advice.
fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<f64> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|s| s.is_finite() && *s >= 0.0)
}

/// A chat client backed by an Azure AI Foundry persistent agent.
pub struct AzureAIAgentClient {
    inner: Arc<Inner>,
}

struct Inner {
    http: reqwest::Client,
    endpoint: String,
    api_version: String,
    credential: Arc<dyn TokenCredential>,
    scope: String,
    model_id: Option<String>,
    agent_name: Option<String>,
    agent_description: Option<String>,
    default_thread_id: Option<String>,
    should_cleanup_agent: bool,
    /// The active agent id: pre-set for an existing agent, or filled in after
    /// auto-creating one.
    agent_id: Mutex<Option<String>>,
    /// Whether this client created the agent (and so should delete it).
    agent_created: AtomicBool,
    /// Cached agent definition (`GET .../assistants/{id}`, or the body
    /// returned when auto-creating), used to replay an agent's own
    /// tools/instructions/tool_resources onto every run — see
    /// [`load_agent_definition_if_needed`](AzureAIAgentClient::load_agent_definition_if_needed).
    agent_definition: Mutex<Option<Value>>,
}

impl Clone for AzureAIAgentClient {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl std::fmt::Debug for AzureAIAgentClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AzureAIAgentClient")
            .field("endpoint", &self.inner.endpoint)
            .field("api_version", &self.inner.api_version)
            .field("model_id", &self.inner.model_id)
            .field("agent_id", &*self.inner.agent_id.lock().unwrap())
            .finish_non_exhaustive()
    }
}

impl AzureAIAgentClient {
    /// Create a client that auto-creates a transient agent for `model` on first
    /// use and deletes it on [`close`](Self::close).
    pub fn new(
        endpoint: impl Into<String>,
        model: impl Into<String>,
        credential: Arc<dyn TokenCredential>,
    ) -> Self {
        Self::build(endpoint.into(), Some(model.into()), None, credential)
    }

    /// Create a client that targets an existing (persistent) agent by id. The
    /// agent is never deleted by this client.
    pub fn with_existing_agent(
        endpoint: impl Into<String>,
        agent_id: impl Into<String>,
        credential: Arc<dyn TokenCredential>,
    ) -> Self {
        Self::build(endpoint.into(), None, Some(agent_id.into()), credential)
    }

    fn build(
        endpoint: String,
        model_id: Option<String>,
        agent_id: Option<String>,
        credential: Arc<dyn TokenCredential>,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                http: reqwest::Client::new(),
                endpoint,
                api_version: DEFAULT_API_VERSION.to_string(),
                credential,
                scope: AI_FOUNDRY_SCOPE.to_string(),
                model_id,
                agent_name: None,
                agent_description: None,
                default_thread_id: None,
                should_cleanup_agent: true,
                agent_id: Mutex::new(agent_id),
                agent_created: AtomicBool::new(false),
                agent_definition: Mutex::new(None),
            }),
        }
    }

    /// Mutable access to `Inner` for the builder methods. `Inner` is not
    /// `Clone` (it holds a `Mutex`/`AtomicBool`), so `Arc::make_mut` is
    /// unavailable; builders are chained onto a freshly-constructed, uniquely
    /// owned client, where `Arc::get_mut` always succeeds.
    fn inner_mut(&mut self) -> &mut Inner {
        Arc::get_mut(&mut self.inner)
            .expect("builder methods must be called before the client is cloned or shared")
    }

    /// Override the API version (default [`DEFAULT_API_VERSION`]).
    pub fn with_api_version(mut self, api_version: impl Into<String>) -> Self {
        self.inner_mut().api_version = api_version.into();
        self
    }

    /// Override the Entra ID token scope (default [`AI_FOUNDRY_SCOPE`]).
    pub fn with_scope(mut self, scope: impl Into<String>) -> Self {
        self.inner_mut().scope = scope.into();
        self
    }

    /// Set the name used when auto-creating an agent (default `"UnnamedAgent"`).
    pub fn with_agent_name(mut self, name: impl Into<String>) -> Self {
        self.inner_mut().agent_name = Some(name.into());
        self
    }

    /// Set the description used when auto-creating an agent.
    pub fn with_agent_description(mut self, description: impl Into<String>) -> Self {
        self.inner_mut().agent_description = Some(description.into());
        self
    }

    /// Set the default thread (conversation) id used when a request does not
    /// carry a `conversation_id`.
    pub fn with_thread_id(mut self, thread_id: impl Into<String>) -> Self {
        self.inner_mut().default_thread_id = Some(thread_id.into());
        self
    }

    /// Whether an auto-created agent is deleted on [`close`](Self::close)
    /// (default `true`).
    pub fn with_cleanup_agent(mut self, cleanup: bool) -> Self {
        self.inner_mut().should_cleanup_agent = cleanup;
        self
    }

    /// The agent id, once known (pre-set for an existing agent, or filled in
    /// after auto-creation).
    pub fn agent_id(&self) -> Option<String> {
        self.inner.agent_id.lock().unwrap().clone()
    }

    /// The project endpoint this client targets.
    pub fn endpoint(&self) -> &str {
        &self.inner.endpoint
    }

    // -- URL + HTTP plumbing ------------------------------------------------

    fn url(&self, path: &str) -> String {
        let base = self.inner.endpoint.trim_end_matches('/');
        let sep = if path.contains('?') { '&' } else { '?' };
        format!("{base}/{path}{sep}api-version={}", self.inner.api_version)
    }

    async fn authorized(
        &self,
        builder: reqwest::RequestBuilder,
    ) -> Result<reqwest::RequestBuilder> {
        let token = self
            .inner
            .credential
            .get_token_for_scope(&self.inner.scope)
            .await?;
        Ok(builder.bearer_auth(token))
    }

    async fn check_status(&self, resp: reqwest::Response) -> Result<reqwest::Response> {
        if resp.status().is_success() {
            return Ok(resp);
        }
        let status = resp.status();
        let retry_after = parse_retry_after(resp.headers());
        let text = resp.text().await.unwrap_or_default();
        Err(Error::service_status(
            status.as_u16(),
            format!("Azure AI API error {status}: {text}"),
            retry_after,
        ))
    }

    async fn post_json(&self, path: &str, body: &Value) -> Result<Value> {
        let req = self
            .authorized(self.inner.http.post(self.url(path)).json(body))
            .await?;
        let resp = req
            .send()
            .await
            .map_err(|e| Error::service(format!("request failed: {e}")))?;
        let resp = self.check_status(resp).await?;
        resp.json()
            .await
            .map_err(|e| Error::service(format!("invalid response json: {e}")))
    }

    async fn get_json(&self, path: &str) -> Result<Value> {
        let req = self.authorized(self.inner.http.get(self.url(path))).await?;
        let resp = req
            .send()
            .await
            .map_err(|e| Error::service(format!("request failed: {e}")))?;
        let resp = self.check_status(resp).await?;
        resp.json()
            .await
            .map_err(|e| Error::service(format!("invalid response json: {e}")))
    }

    async fn delete_path(&self, path: &str) -> Result<()> {
        let req = self
            .authorized(self.inner.http.delete(self.url(path)))
            .await?;
        let resp = req
            .send()
            .await
            .map_err(|e| Error::service(format!("request failed: {e}")))?;
        self.check_status(resp).await?;
        Ok(())
    }

    async fn post_stream(&self, path: &str, body: &Value) -> Result<reqwest::Response> {
        let req = self
            .authorized(
                self.inner
                    .http
                    .post(self.url(path))
                    .header(reqwest::header::ACCEPT, "text/event-stream")
                    .json(body),
            )
            .await?;
        let resp = req
            .send()
            .await
            .map_err(|e| Error::service(format!("request failed: {e}")))?;
        self.check_status(resp).await
    }

    // -- Operations ---------------------------------------------------------

    /// Create an agent (`POST …/assistants`). Public for parity/testing.
    pub async fn create_agent(&self, body: &Value) -> Result<Value> {
        self.post_json("assistants", body).await
    }

    /// Fetch an agent's definition (`GET …/assistants/{id}`).
    pub async fn get_agent(&self, agent_id: &str) -> Result<Value> {
        self.get_json(&format!("assistants/{agent_id}")).await
    }

    /// Delete an agent (`DELETE …/assistants/{id}`).
    pub async fn delete_agent(&self, agent_id: &str) -> Result<()> {
        self.delete_path(&format!("assistants/{agent_id}")).await
    }

    /// Delete a thread (`DELETE …/threads/{id}`).
    pub async fn delete_thread(&self, thread_id: &str) -> Result<()> {
        self.delete_path(&format!("threads/{thread_id}")).await
    }

    async fn ensure_agent(
        &self,
        options: &ChatOptions,
        instructions: Option<&str>,
    ) -> Result<String> {
        {
            let guard = self.inner.agent_id.lock().unwrap();
            if let Some(id) = guard.as_ref() {
                return Ok(id.clone());
            }
        }
        let model = options
            .model_id
            .as_deref()
            .or(self.inner.model_id.as_deref())
            .ok_or_else(|| {
                Error::Configuration(
                    "a model deployment name is required to auto-create an agent".into(),
                )
            })?;
        let name = self.inner.agent_name.as_deref().or(Some("UnnamedAgent"));
        let body = convert::build_agent_body(
            model,
            name,
            self.inner.agent_description.as_deref(),
            instructions,
            options,
        )?;
        let created = self.create_agent(&body).await?;
        let id = created
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::service("agent create response missing 'id'"))?
            .to_string();
        *self.inner.agent_id.lock().unwrap() = Some(id.clone());
        self.inner.agent_created.store(true, Ordering::SeqCst);
        // Cache the just-created agent's own definition so it gets replayed
        // on later turns the same way an existing (persistent) agent's does
        // (see `load_agent_definition_if_needed`), instead of an extra GET.
        *self.inner.agent_definition.lock().unwrap() = Some(created);
        Ok(id)
    }

    /// Fetch (and cache) the active agent's own definition, mirroring the
    /// Python client's `_load_agent_definition_if_needed`
    /// (`_chat_client.py:751-755`): a no-op until an agent id is known (i.e.
    /// before an auto-created agent exists yet), fetched at most once per
    /// agent id and reused after that. [`ensure_agent`](Self::ensure_agent)
    /// seeds this cache directly from its create response, so a freshly
    /// auto-created agent never needs an extra `GET` either.
    async fn load_agent_definition_if_needed(&self) -> Result<Option<Value>> {
        let id = { self.inner.agent_id.lock().unwrap().clone() };
        let Some(id) = id else {
            return Ok(None);
        };
        {
            let cached = self.inner.agent_definition.lock().unwrap();
            if let Some(def) = cached.as_ref() {
                return Ok(Some(def.clone()));
            }
        }
        let def = self.get_agent(&id).await?;
        *self.inner.agent_definition.lock().unwrap() = Some(def.clone());
        Ok(Some(def))
    }

    async fn create_thread(&self, options: &ChatOptions) -> Result<String> {
        let mut body = Map::new();
        if let Some(meta) = &options.metadata {
            body.insert("metadata".into(), json!(meta));
        }
        let created = self.post_json("threads", &Value::Object(body)).await?;
        created
            .get("id")
            .and_then(Value::as_str)
            .map(String::from)
            .ok_or_else(|| Error::service("thread create response missing 'id'"))
    }

    async fn add_messages(&self, thread_id: &str, messages: &[Value]) -> Result<()> {
        for msg in messages {
            self.post_json(&format!("threads/{thread_id}/messages"), msg)
                .await?;
        }
        Ok(())
    }

    async fn list_run_messages(&self, thread_id: &str, run_id: &str) -> Result<Value> {
        self.get_json(&format!(
            "threads/{thread_id}/messages?run_id={run_id}&order=asc"
        ))
        .await
    }

    async fn poll_to_terminal(
        &self,
        thread_id: &str,
        run_id: &str,
        mut run: Value,
    ) -> Result<Value> {
        for _ in 0..MAX_POLLS {
            let status = convert::run_status(&run).unwrap_or("");
            if convert::is_terminal_status(status) {
                return Ok(run);
            }
            tokio::time::sleep(POLL_INTERVAL).await;
            run = self
                .get_json(&format!("threads/{thread_id}/runs/{run_id}"))
                .await?;
        }
        Err(Error::service(
            "Azure AI run did not reach a terminal state before the poll timeout",
        ))
    }

    /// Delete an auto-created agent (no-op for an existing agent or when
    /// cleanup is disabled). Rust has no async `Drop`, so call this explicitly
    /// when done with a client that auto-created its agent.
    pub async fn close(&self) -> Result<()> {
        if self.inner.agent_created.load(Ordering::SeqCst) && self.inner.should_cleanup_agent {
            let id = self.inner.agent_id.lock().unwrap().clone();
            if let Some(id) = id {
                self.delete_agent(&id).await?;
                *self.inner.agent_id.lock().unwrap() = None;
                self.inner.agent_created.store(false, Ordering::SeqCst);
                *self.inner.agent_definition.lock().unwrap() = None;
            }
        }
        Ok(())
    }

    // -- Shared request setup ----------------------------------------------

    fn thread_from_options(&self, options: &ChatOptions) -> Option<String> {
        options
            .conversation_id
            .clone()
            .or_else(|| self.inner.default_thread_id.clone())
    }

    fn model_for<'a>(&'a self, options: &'a ChatOptions) -> Option<&'a str> {
        options
            .model_id
            .as_deref()
            .or(self.inner.model_id.as_deref())
    }

    /// Build a completed/requires-action [`ChatResponse`] from a terminal run.
    async fn response_from_run(&self, thread_id: &str, run: Value) -> Result<ChatResponse> {
        let status = convert::run_status(&run).unwrap_or("").to_string();
        let run_id = run
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();

        let mut resp = ChatResponse {
            conversation_id: Some(thread_id.to_string()),
            response_id: Some(run_id.clone()),
            usage_details: convert::parse_usage(&run),
            finish_reason: convert::finish_reason_for(&status),
            ..Default::default()
        };

        match status.as_str() {
            "requires_action" => {
                let contents = convert::required_action_contents(&run, &run_id);
                let mut m = ChatMessage::with_contents(Role::assistant(), contents);
                m.message_id = Some(run_id);
                resp.messages.push(m);
            }
            "completed" => {
                let list = self.list_run_messages(thread_id, &run_id).await?;
                let text = convert::assistant_text_from_messages(&list);
                let mut m = ChatMessage::assistant(text);
                m.message_id = Some(run_id);
                resp.messages.push(m);
            }
            "failed" => return Err(Error::service(convert::last_error_message(&run))),
            other => {
                return Err(Error::service(format!(
                    "Azure AI run ended in status '{other}'"
                )))
            }
        }
        Ok(resp)
    }
}

fn combined_instructions(options: &ChatOptions, prepared: &PreparedMessages) -> Option<String> {
    match (options.instructions.clone(), prepared.instructions.clone()) {
        (Some(a), Some(b)) => Some(format!("{a}\n{b}")),
        (Some(a), None) => Some(a),
        (None, b) => b,
    }
}

#[async_trait::async_trait]
impl ChatClient for AzureAIAgentClient {
    async fn get_response(
        &self,
        messages: Vec<ChatMessage>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        let prepared = convert::prepare_messages(&messages);
        let instructions = combined_instructions(&options, &prepared);
        let thread_id = self.thread_from_options(&options);

        // Tool-result submission path: answer an active run's `requires_action`.
        if let Some(run_id) = prepared.run_id.clone() {
            let thread_id = thread_id.ok_or_else(|| {
                Error::service("conversation_id is required to submit tool outputs")
            })?;
            let body =
                convert::build_submit_body(&prepared.tool_outputs, &prepared.tool_approvals, false);
            let run = self
                .post_json(
                    &format!("threads/{thread_id}/runs/{run_id}/submit_tool_outputs"),
                    &body,
                )
                .await?;
            let run = self.poll_to_terminal(&thread_id, &run_id, run).await?;
            return self.response_from_run(&thread_id, run).await;
        }

        // Fresh-run path.
        let agent_definition = self.load_agent_definition_if_needed().await?;
        let agent_id = self.ensure_agent(&options, instructions.as_deref()).await?;
        let thread_id = match thread_id {
            Some(t) => t,
            None => self.create_thread(&options).await?,
        };
        self.add_messages(&thread_id, &prepared.messages).await?;
        let body = convert::build_run_body(
            &agent_id,
            self.model_for(&options),
            instructions.as_deref(),
            &options,
            agent_definition.as_ref(),
            false,
        )?;
        let run = self
            .post_json(&format!("threads/{thread_id}/runs"), &body)
            .await?;
        let run_id = run
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let run = self.poll_to_terminal(&thread_id, &run_id, run).await?;
        self.response_from_run(&thread_id, run).await
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<ChatMessage>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let prepared = convert::prepare_messages(&messages);
        let instructions = combined_instructions(&options, &prepared);
        let thread_id = self.thread_from_options(&options);

        // Tool-result submission path (streamed).
        if let Some(run_id) = prepared.run_id.clone() {
            let thread_id = thread_id.ok_or_else(|| {
                Error::service("conversation_id is required to submit tool outputs")
            })?;
            let body =
                convert::build_submit_body(&prepared.tool_outputs, &prepared.tool_approvals, true);
            let resp = self
                .post_stream(
                    &format!("threads/{thread_id}/runs/{run_id}/submit_tool_outputs"),
                    &body,
                )
                .await?;
            return Ok(sse::parse_agent_sse_stream(resp, thread_id).boxed());
        }

        // Fresh-run path (streamed).
        let agent_definition = self.load_agent_definition_if_needed().await?;
        let agent_id = self.ensure_agent(&options, instructions.as_deref()).await?;
        let thread_id = match thread_id {
            Some(t) => t,
            None => self.create_thread(&options).await?,
        };
        self.add_messages(&thread_id, &prepared.messages).await?;
        let body = convert::build_run_body(
            &agent_id,
            self.model_for(&options),
            instructions.as_deref(),
            &options,
            agent_definition.as_ref(),
            true,
        )?;
        let resp = self
            .post_stream(&format!("threads/{thread_id}/runs"), &body)
            .await?;
        Ok(sse::parse_agent_sse_stream(resp, thread_id).boxed())
    }

    fn model_id(&self) -> Option<&str> {
        self.inner.model_id.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_azure::StaticTokenCredential;

    fn client() -> AzureAIAgentClient {
        AzureAIAgentClient::new(
            "https://proj.services.ai.azure.com/",
            "gpt-4o",
            Arc::new(StaticTokenCredential::new("tok")),
        )
    }

    #[test]
    fn url_appends_api_version() {
        let c = client();
        assert_eq!(
            c.url("threads"),
            "https://proj.services.ai.azure.com/threads?api-version=2025-05-01"
        );
        // A path that already has a query string gets `&api-version`.
        assert_eq!(
            c.url("threads/t1/messages?run_id=r1&order=asc"),
            "https://proj.services.ai.azure.com/threads/t1/messages?run_id=r1&order=asc&api-version=2025-05-01"
        );
    }

    #[test]
    fn with_api_version_overrides_default() {
        let c = client().with_api_version("2025-05-15-preview");
        assert!(c.url("threads").ends_with("api-version=2025-05-15-preview"));
    }

    #[test]
    fn existing_agent_is_known_and_not_created() {
        let c = AzureAIAgentClient::with_existing_agent(
            "https://proj.services.ai.azure.com",
            "asst_1",
            Arc::new(StaticTokenCredential::new("tok")),
        );
        assert_eq!(c.agent_id().as_deref(), Some("asst_1"));
        assert!(!c.inner.agent_created.load(Ordering::SeqCst));
    }

    #[test]
    fn model_id_reflects_configured_deployment() {
        assert_eq!(client().model_id(), Some("gpt-4o"));
    }

    #[test]
    fn instructions_combine_options_and_system_messages() {
        let prepared = convert::prepare_messages(&[ChatMessage::system("from message")]);
        let options = ChatOptions::new().with_instructions("from options");
        assert_eq!(
            combined_instructions(&options, &prepared).as_deref(),
            Some("from options\nfrom message")
        );
    }
}
