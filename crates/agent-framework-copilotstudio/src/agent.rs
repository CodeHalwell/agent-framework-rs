//! [`CopilotStudioAgent`]: wraps the Copilot Studio Direct-to-Engine API as a
//! local [`SupportsAgentRun`].

use std::sync::Arc;

use async_trait::async_trait;
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, USER_AGENT};

use agent_framework_core::agent::SupportsAgentRun;
use agent_framework_core::error::{Error, Result};
use agent_framework_core::threads::AgentThread;
use agent_framework_core::types::{AgentResponse, IntoMessages, Message, Role};

use crate::activity::{
    build_message_activity_body, build_start_conversation_body, parse_activities, WireActivity,
};
use crate::auth::TokenProvider;
use crate::settings::CopilotStudioConnectionSettings;

const EVENT_STREAM_ACCEPT: &str = "text/event-stream";
const D2E_USER_AGENT: &str = concat!(
    "agent-framework-rs-copilotstudio/",
    env!("CARGO_PKG_VERSION")
);

/// A Microsoft Copilot Studio agent, reached via the Direct-to-Engine (D2E)
/// API. Wraps a [`CopilotStudioConnectionSettings`] + [`TokenProvider`] so a
/// published (or prebuilt) Copilot Studio agent can be used anywhere the
/// framework expects a local [`SupportsAgentRun`].
///
/// See the crate docs for the exact wire protocol, the auth burden this port
/// pushes onto callers, and how conversation continuity here diverges
/// (improves) on the Python reference.
#[derive(Clone)]
pub struct CopilotStudioAgent {
    id: String,
    name: Option<String>,
    description: Option<String>,
    connection: Arc<CopilotStudioConnectionSettings>,
    token_provider: Arc<dyn TokenProvider>,
    http: reqwest::Client,
}

impl CopilotStudioAgent {
    /// Wrap a [`CopilotStudioConnectionSettings`] + [`TokenProvider`] as an
    /// agent. Prefer [`CopilotStudioConnectionSettings::from_settings`]
    /// (fed by [`CopilotStudioSettings::from_env`](crate::settings::CopilotStudioSettings::from_env))
    /// when you want Python's `COPILOTSTUDIOAGENT__*`-env-var-driven
    /// construction.
    pub fn new(
        connection: CopilotStudioConnectionSettings,
        token_provider: impl TokenProvider + 'static,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name: None,
            description: None,
            connection: Arc::new(connection),
            token_provider: Arc::new(token_provider),
            http: reqwest::Client::new(),
        }
    }

    /// Override the agent id (defaults to a random UUID).
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Set the agent's display name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the agent description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// The agent description, if any.
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// The connection settings this agent was built with (D2E environment,
    /// cloud, agent type, ...).
    pub fn connection(&self) -> &CopilotStudioConnectionSettings {
        &self.connection
    }

    /// Ergonomic run without an explicit thread (mirrors
    /// `Agent::run_once`): the conversation starts fresh every call,
    /// since no [`AgentThread`] is carried across calls to persist the
    /// Direct-to-Engine conversation id.
    pub async fn run_once(&self, messages: impl IntoMessages) -> Result<AgentResponse> {
        self.run(messages.into_messages(), None).await
    }

    async fn bearer_header(&self) -> Result<String> {
        let token = self.token_provider.get_token().await?;
        Ok(format!("Bearer {token}"))
    }

    /// POST `body` to `url`, requesting `text/event-stream`, and parse the
    /// resulting activities (SSE or JSON array — see
    /// [`crate::activity::parse_activities`]).
    async fn post_activities(
        &self,
        url: &str,
        body: serde_json::Value,
    ) -> Result<Vec<WireActivity>> {
        let auth = self.bearer_header().await?;
        let resp = self
            .http
            .post(url)
            .header(AUTHORIZATION, auth)
            .header(CONTENT_TYPE, "application/json")
            .header(ACCEPT, EVENT_STREAM_ACCEPT)
            .header(USER_AGENT, D2E_USER_AGENT)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::service(format!("Copilot Studio request to {url} failed: {e}")))?;
        let status = resp.status();
        let text = resp.text().await.map_err(|e| {
            Error::service(format!("failed reading Copilot Studio response body: {e}"))
        })?;
        if !status.is_success() {
            return Err(Error::service_status(status.as_u16(), text, None));
        }
        Ok(parse_activities(&text))
    }

    /// Start a new Direct-to-Engine conversation and return its conversation
    /// id. Mirrors `CopilotStudioAgent._start_new_conversation` /
    /// `CopilotClient.start_conversation`: POSTs
    /// `{"emitStartConversationEvent": true}` to the conversation-less URL,
    /// then scans the returned activities for the *last* one carrying a
    /// `conversation.id` (mirroring the Python loop, which keeps overwriting
    /// `conversation_id` for every matching activity rather than stopping at
    /// the first).
    async fn start_conversation(&self) -> Result<String> {
        let url = self.connection.conversation_url(None)?;
        let body = build_start_conversation_body(true);
        let activities = self.post_activities(&url, body).await?;
        activities
            .iter()
            .rev()
            .find_map(|a| a.conversation.as_ref().map(|c| c.id.clone()))
            .ok_or_else(|| Error::service("Failed to start a new conversation."))
    }
}

#[async_trait]
impl SupportsAgentRun for CopilotStudioAgent {
    async fn run(
        &self,
        messages: Vec<Message>,
        thread: Option<&mut AgentThread>,
    ) -> Result<AgentResponse> {
        // Mirrors agent-framework-a2a's A2AAgent: only the newest message is
        // sent — with real conversation-id continuity (see below and the
        // crate docs), the server already has everything earlier. Python's
        // CopilotStudioAgent instead joins *every* message passed to this
        // call's `messages` argument with "\n" into one question string;
        // this diverges deliberately (see crate docs).
        let last = messages.last().ok_or_else(|| {
            Error::AgentExecution("CopilotStudioAgent::run requires at least one message".into())
        })?;
        let question = last.text();

        let mut owned_thread;
        let thread: &mut AgentThread = match thread {
            Some(t) => t,
            None => {
                owned_thread = self.get_new_thread();
                &mut owned_thread
            }
        };

        // Continuity: reuse an existing conversation id on this thread;
        // only start a new Direct-to-Engine conversation the first time the
        // thread is used. Python's CopilotStudioAgent.run unconditionally
        // calls `_start_new_conversation()` on *every* call, discarding any
        // conversation id already on the thread — seemingly not intentional
        // multi-turn support, since it throws away prior context every
        // time. This port fixes that the same way agent-framework-a2a's
        // A2AAgent does for contextId/taskId: persist and reuse.
        let conversation_id = match thread.service_thread_id() {
            Some(id) => id.to_string(),
            None => {
                let id = self.start_conversation().await?;
                // Best-effort: only fails if `thread` already owns a local
                // message store (mutually exclusive with a service thread
                // id), which a freshly-created `AgentThread` never does.
                let _ = thread.set_service_thread_id(id.clone());
                id
            }
        };

        let url = self.connection.conversation_url(Some(&conversation_id))?;
        let body = build_message_activity_body(&question, &conversation_id);
        let activities = self.post_activities(&url, body).await?;

        // Mirrors `_process_activities(activities, streaming=False)`: only
        // `type == "message"` activities become response messages; `typing`/
        // `trace`/anything else is skipped.
        let mut response_messages: Vec<Message> = Vec::new();
        for activity in &activities {
            if activity.activity_type != "message" {
                continue;
            }
            let Some(text) = activity.text.as_ref().filter(|t| !t.is_empty()) else {
                continue;
            };
            let mut message = Message::new(Role::assistant(), text.clone());
            message.message_id = activity.id.clone();
            message.author_name = activity.from.as_ref().and_then(|f| f.name.clone());
            response_messages.push(message);
        }

        let response_id = response_messages.first().and_then(|m| m.message_id.clone());

        let mut response = AgentResponse {
            messages: response_messages,
            response_id,
            ..Default::default()
        };
        if let Some(name) = &self.name {
            for m in &mut response.messages {
                if m.author_name.is_none() {
                    m.author_name = Some(name.clone());
                }
            }
        }
        Ok(response)
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::StaticTokenProvider;

    fn agent() -> CopilotStudioAgent {
        CopilotStudioAgent::new(
            CopilotStudioConnectionSettings::new("52d1e846e0804341a63f58f2ab72fb28", "test-schema"),
            StaticTokenProvider::new("test-token"),
        )
    }

    #[test]
    fn from_url_style_constructor_sets_random_id() {
        let agent = agent();
        assert!(!agent.id().is_empty());
        assert!(agent.name().is_none());
    }

    #[test]
    fn with_id_and_name_and_description_override_defaults() {
        let agent = agent()
            .with_id("fixed-id")
            .with_name("my-copilot")
            .with_description("Helps with things");
        assert_eq!(agent.id(), "fixed-id");
        assert_eq!(agent.name(), Some("my-copilot"));
        assert_eq!(agent.description(), Some("Helps with things"));
    }

    #[test]
    fn display_name_falls_back_to_id_when_unset() {
        let unnamed = agent().with_id("fixed-id");
        assert_eq!(unnamed.display_name(), "fixed-id");
        let named = agent().with_name("n");
        assert_eq!(named.display_name(), "n");
    }

    #[tokio::test]
    async fn run_with_no_messages_errors_without_any_network_access() {
        // If this didn't fail fast, it would hang trying to reach a fake
        // Power Platform host.
        let err = agent().run(Vec::new(), None).await.unwrap_err();
        assert!(matches!(err, Error::AgentExecution(_)));
    }
}
