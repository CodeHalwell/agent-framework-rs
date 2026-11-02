//! A chat client bound to an agent that already exists in a Foundry project.
//!
//! Port of upstream's `RawFoundryAgentChatClient` / `FoundryAgent(agent_name=...)`
//! (.NET `AIProjectClient.AsAIAgent`). Where [`FoundryAgent`](crate::FoundryAgent)
//! realises a Prompt Agent on the client (sending its model, instructions
//! and tools with every request), [`FoundryAgentChatClient`] calls a Prompt
//! or Hosted Agent defined in the service: every Responses request carries
//! an `agent_reference` (`{"type": "agent_reference", "name", "version"}`)
//! and leaves the rest to the agent's server-side definition.
//!
//! Per request, as upstream:
//!
//! - `agent_reference` is added unless the caller supplied one;
//! - `model` is dropped (the agent selects it) unless the caller set one
//!   on a continuation, and always when `agent_session_id` is sent;
//! - tool declarations, `tool_choice` and `parallel_tool_calls` are dropped,
//!   because the service rejects them alongside an agent reference ("Not
//!   allowed when agent is specified"). Local function tools still run: the
//!   agent's function-invocation loop dispatches the calls the service
//!   makes. Non-function tools are refused, since they can only be defined
//!   on the service.
//!
//! ```no_run
//! use std::sync::Arc;
//! use agent_framework_foundry::{FoundryAgentChatClient, FOUNDRY_SCOPE};
//!
//! # async fn demo() -> agent_framework_core::error::Result<()> {
//! let credential = Arc::new(agent_framework_azure::DefaultAzureCredential::new(FOUNDRY_SCOPE));
//! let client = FoundryAgentChatClient::with_token_credential(
//!     "https://my-project.services.ai.azure.com/api/projects/demo",
//!     "support-agent",
//!     Some("3".into()),
//!     credential,
//! );
//! let agent = client.as_agent().build();
//! let answer = agent.run_once("Where is my order?").await?;
//! # let _ = answer;
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use agent_framework_azure::responses::AzureOpenAIResponsesClient;
use agent_framework_azure::TokenCredential;
use agent_framework_core::agent::{Agent, AgentBuilder};
use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::error::{Error, Result};
use agent_framework_core::tools::ToolKind;
use agent_framework_core::types::{ChatOptions, ChatResponse, Message, ToolMode};
use async_trait::async_trait;
use serde_json::{json, Map, Value};

/// A [`ChatClient`] calling an agent defined in a Foundry project. See the
/// [module docs](self).
#[derive(Clone)]
pub struct FoundryAgentChatClient {
    inner: AzureOpenAIResponsesClient,
    endpoint: String,
    agent_name: String,
    agent_version: Option<String>,
}

impl std::fmt::Debug for FoundryAgentChatClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FoundryAgentChatClient")
            .field("endpoint", &self.endpoint)
            .field("agent_name", &self.agent_name)
            .field("agent_version", &self.agent_version)
            .finish_non_exhaustive()
    }
}

/// The `agent_reference` request field.
pub fn agent_reference(agent_name: &str, agent_version: Option<&str>) -> Value {
    let mut reference = json!({"name": agent_name, "type": "agent_reference"});
    if let Some(version) = agent_version.filter(|v| !v.is_empty()) {
        reference["version"] = json!(version);
    }
    reference
}

/// Rewrite a Responses body for an agent call; see the [module docs](self).
pub fn apply_agent_request(
    body: &mut Map<String, Value>,
    options: &ChatOptions,
    agent_name: &str,
    agent_version: Option<&str>,
) {
    body.entry("agent_reference")
        .or_insert_with(|| agent_reference(agent_name, agent_version));
    let has_session = body.contains_key("agent_session_id");
    if has_session || (options.conversation_id.is_none() && options.model.is_none()) {
        body.remove("model");
    }
    let had_tools = body.remove("tools").is_some();
    body.remove("tool_choice");
    body.remove("parallel_tool_calls");
    let unrestricted_auto = matches!(options.tool_choice, None | Some(ToolMode::Auto));
    if !unrestricted_auto || options.allow_multiple_tool_calls.is_some() {
        tracing::warn!(
            agent = %agent_name,
            "the Foundry agent owns tool selection server-side; tool_choice and allow_multiple_tool_calls are ignored"
        );
    }
    if had_tools {
        tracing::debug!(
            agent = %agent_name,
            "tool declarations are omitted when an agent is specified; local tools are used only for dispatch"
        );
    }
}

impl FoundryAgentChatClient {
    fn build(
        inner: AzureOpenAIResponsesClient,
        endpoint: String,
        agent_name: String,
        agent_version: Option<String>,
    ) -> Self {
        let (name, version) = (agent_name.clone(), agent_version.clone());
        let inner = inner
            .without_api_version()
            .without_implicit_encrypted_reasoning()
            .with_http_client(crate::http_client())
            .with_body_transform(move |body, options| {
                apply_agent_request(body, options, &name, version.as_deref())
            });
        Self {
            inner,
            endpoint,
            agent_name,
            agent_version,
        }
    }

    /// A client for agent `agent_name` (and `agent_version`, required for a
    /// Prompt Agent) in the project at `project_endpoint`, with an API key.
    pub fn new(
        project_endpoint: impl Into<String>,
        agent_name: impl Into<String>,
        agent_version: Option<String>,
        api_key: impl Into<String>,
    ) -> Self {
        let endpoint = project_endpoint.into();
        // The model is the agent's; the deployment name is never sent.
        let inner = AzureOpenAIResponsesClient::new(endpoint.clone(), "", api_key);
        Self::build(inner, endpoint, agent_name.into(), agent_version)
    }

    /// As [`new`](Self::new), authenticating with Entra ID. The credential
    /// should be scoped to [`FOUNDRY_SCOPE`](crate::FOUNDRY_SCOPE).
    pub fn with_token_credential(
        project_endpoint: impl Into<String>,
        agent_name: impl Into<String>,
        agent_version: Option<String>,
        credential: Arc<dyn TokenCredential>,
    ) -> Self {
        let endpoint = project_endpoint.into();
        let inner =
            AzureOpenAIResponsesClient::with_token_credential(endpoint.clone(), "", credential);
        Self::build(inner, endpoint, agent_name.into(), agent_version)
    }

    /// From `FOUNDRY_PROJECT_ENDPOINT` (or `FOUNDRY_ENDPOINT`),
    /// `FOUNDRY_AGENT_NAME` and optional `FOUNDRY_AGENT_VERSION`, with
    /// `FOUNDRY_API_KEY` or else a `DefaultAzureCredential`.
    pub fn from_env() -> Result<Self> {
        let get = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        let endpoint = get("FOUNDRY_PROJECT_ENDPOINT")
            .or_else(|| get("FOUNDRY_ENDPOINT"))
            .ok_or_else(|| {
                Error::Configuration(
                    "Either 'project_endpoint' or FOUNDRY_PROJECT_ENDPOINT is required.".into(),
                )
            })?;
        let name = get("FOUNDRY_AGENT_NAME").ok_or_else(|| {
            Error::Configuration(
                "Agent name is required. Set via 'agent_name' or the 'FOUNDRY_AGENT_NAME' environment variable."
                    .into(),
            )
        })?;
        let version = get("FOUNDRY_AGENT_VERSION");
        Ok(match get("FOUNDRY_API_KEY") {
            Some(key) => Self::new(endpoint, name, version, key),
            None => Self::with_token_credential(
                endpoint,
                name,
                version,
                Arc::new(agent_framework_azure::DefaultAzureCredential::new(
                    crate::FOUNDRY_SCOPE,
                )),
            ),
        })
    }

    /// Override the base URL requests are built against.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.inner = self.inner.with_base_url(base_url);
        self
    }

    /// The agent's name.
    pub fn agent_name(&self) -> &str {
        &self.agent_name
    }

    /// The agent's version, if pinned.
    pub fn agent_version(&self) -> Option<&str> {
        self.agent_version.as_deref()
    }

    /// An [`AgentBuilder`] over this client, named after the agent. Add local
    /// function tools to dispatch the calls the service agent makes.
    pub fn as_agent(&self) -> AgentBuilder {
        Agent::builder(self.clone()).name(self.agent_name.clone())
    }

    fn check_tools(&self, options: &ChatOptions) -> Result<()> {
        if let Some(tool) = options
            .tools
            .iter()
            .find(|t| !matches!(t.kind, ToolKind::Function))
        {
            return Err(Error::Configuration(format!(
                "Only function tools are accepted for Foundry agents, got '{}'. Hosted tools, MCP tools \
                 and the like must be defined on the Foundry agent definition in the service.",
                tool.name
            )));
        }
        Ok(())
    }
}

#[async_trait]
impl ChatClient for FoundryAgentChatClient {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        self.check_tools(&options)?;
        self.inner.get_response(messages, options).await
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        self.check_tools(&options)?;
        self.inner.get_streaming_response(messages, options).await
    }

    fn model(&self) -> Option<&str> {
        None
    }

    fn service_url(&self) -> Option<&str> {
        Some(&self.endpoint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body() -> Map<String, Value> {
        json!({"model": "", "input": [], "tools": [{"type": "function"}], "tool_choice": "auto", "parallel_tool_calls": true})
            .as_object()
            .unwrap()
            .clone()
    }

    #[test]
    fn adds_the_reference_and_strips_model_and_tools() {
        let mut b = body();
        apply_agent_request(&mut b, &ChatOptions::default(), "support", Some("3"));
        assert_eq!(
            b["agent_reference"],
            json!({"name": "support", "type": "agent_reference", "version": "3"})
        );
        for gone in ["model", "tools", "tool_choice", "parallel_tool_calls"] {
            assert!(!b.contains_key(gone), "{gone}");
        }
    }

    #[test]
    fn keeps_a_caller_model_on_a_continuation_and_a_caller_reference() {
        let mut b = body();
        b.insert("agent_reference".into(), json!({"name": "other"}));
        let options = ChatOptions {
            conversation_id: Some("resp_1".into()),
            model: Some("gpt-x".into()),
            ..Default::default()
        };
        apply_agent_request(&mut b, &options, "support", None);
        assert_eq!(b["agent_reference"], json!({"name": "other"}));
        assert!(b.contains_key("model"));
        b.insert("agent_session_id".into(), json!("s"));
        apply_agent_request(&mut b, &options, "support", None);
        assert!(
            !b.contains_key("model"),
            "a session always strips the model"
        );
    }

    #[test]
    fn a_versionless_reference_has_no_version() {
        assert_eq!(
            agent_reference("hosted", None),
            json!({"name": "hosted", "type": "agent_reference"})
        );
    }
}
