//! [`A2AClient`]: a JSON-RPC 2.0 / HTTP client for the Agent2Agent protocol.

use std::collections::VecDeque;
use std::pin::Pin;
use std::time::Duration;

use futures::{Stream, StreamExt};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, ACCEPT};
use serde::de::DeserializeOwned;
use serde_json::{json, Value};
use tokio::sync::RwLock;

use agent_framework_core::error::{Error, Result};

use crate::protocol;
use crate::types::{
    AgentCard, MessageSendParams, MessageStreamEvent, PushNotificationConfig, SendMessageResult,
    Task, TaskPushNotificationConfig,
};

const WELL_KNOWN_AGENT_CARD_PATH: &str = "/.well-known/agent-card.json";
const WELL_KNOWN_AGENT_JSON_PATH: &str = "/.well-known/agent.json";

/// A boxed stream of `message/stream` events. See
/// [`A2AClient::send_message_stream`].
pub type A2AEventStream = Pin<Box<dyn Stream<Item = Result<MessageStreamEvent>> + Send>>;

/// A JSON-RPC 2.0 / HTTP client for the Agent2Agent protocol: `message/send`,
/// `message/stream`, `tasks/get`, `tasks/cancel`, plus [`AgentCard`]
/// discovery.
///
/// Not `Clone` (matching `agent-framework-mcp`'s `McpStdioTool` /
/// `McpStreamableHttpTool`); wrap in `Arc` (as [`crate::A2AAgent`] does) to
/// share across tasks.
#[derive(Debug)]
pub struct A2AClient {
    http: reqwest::Client,
    /// Origin used for `.well-known` discovery GETs.
    discovery_base: String,
    /// Extra headers (auth, etc.) sent with every request. Set at
    /// construction time via [`Self::with_header`]; not mutated afterward.
    headers: HeaderMap,
    /// The JSON-RPC POST endpoint. Starts as the constructor's URL and is
    /// updated to the discovered/provided card's `url` once a card is known.
    rpc_url: RwLock<String>,
    /// Cached agent card, once known (provided directly, or discovered via
    /// [`Self::get_agent_card`]).
    card: RwLock<Option<AgentCard>>,
}

impl A2AClient {
    /// Create a client that POSTs JSON-RPC requests directly to `url`, and
    /// uses `url`'s origin for `.well-known` discovery if
    /// [`Self::get_agent_card`] is called.
    ///
    /// This works even against a server with no discovery document at all
    /// (`url` is used as the JSON-RPC endpoint either way); call
    /// [`Self::get_agent_card`] only if you want the server's real
    /// name/description/skills/capabilities.
    pub fn from_url(url: impl Into<String>) -> Self {
        let url = url.into();
        Self {
            http: reqwest::Client::new(),
            rpc_url: RwLock::new(url.clone()),
            discovery_base: url,
            headers: HeaderMap::new(),
            card: RwLock::new(None),
        }
    }

    /// Create a client from an already-known [`AgentCard`]: requests are
    /// POSTed to `card.url` and [`Self::get_agent_card`] returns it directly,
    /// without ever fetching anything.
    pub fn from_card(card: AgentCard) -> Self {
        Self {
            http: reqwest::Client::new(),
            rpc_url: RwLock::new(card.url.clone()),
            discovery_base: card.url.clone(),
            headers: HeaderMap::new(),
            card: RwLock::new(Some(card)),
        }
    }

    /// Add a header (e.g. `Authorization`, or a custom API-key header) sent
    /// with every request this client makes, including discovery GETs.
    pub fn with_header(mut self, name: impl AsRef<str>, value: impl AsRef<str>) -> Result<Self> {
        let name = HeaderName::try_from(name.as_ref())
            .map_err(|e| Error::Configuration(format!("invalid A2A header name: {e}")))?;
        let value = HeaderValue::try_from(value.as_ref())
            .map_err(|e| Error::Configuration(format!("invalid A2A header value: {e}")))?;
        self.headers.insert(name, value);
        Ok(self)
    }

    /// Convenience for `with_header("Authorization", "Bearer <token>")`.
    pub fn with_bearer_token(self, token: impl AsRef<str>) -> Result<Self> {
        self.with_header("Authorization", format!("Bearer {}", token.as_ref()))
    }

    /// Override the request timeout (reqwest's default is no timeout).
    pub fn with_timeout(mut self, timeout: Duration) -> Result<Self> {
        self.http = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| Error::Configuration(format!("failed to build A2A http client: {e}")))?;
        Ok(self)
    }

    /// The card, if already known (provided via [`Self::from_card`], or
    /// fetched by a prior [`Self::get_agent_card`] call). Never performs I/O.
    pub async fn cached_agent_card(&self) -> Option<AgentCard> {
        self.card.read().await.clone()
    }

    /// The JSON-RPC endpoint currently in use: the constructor URL, or the
    /// discovered/provided card's `url` if a card is known.
    pub async fn rpc_url(&self) -> String {
        self.rpc_url.read().await.clone()
    }

    /// Fetch (and cache) the agent's [`AgentCard`].
    ///
    /// GETs `{base}/.well-known/agent-card.json` first; if that fails (older
    /// servers, or ones with no discovery document at all), falls back to
    /// `{base}/.well-known/agent.json`. If the resulting card sets
    /// `supportsAuthenticatedExtendedCard`, this then best-effort upgrades to
    /// the extended card via [`Self::get_extended_card`] — on failure (e.g.
    /// missing/invalid auth headers, or the server not actually honoring the
    /// flag), it silently keeps the base card rather than fail discovery
    /// entirely, mirroring the `a2a-sdk` Python package's `Client.get_card`.
    /// On success either way, also updates the JSON-RPC endpoint used by
    /// subsequent calls to the card's `url`.
    ///
    /// Idempotent: once a card is known (from either path, or from
    /// [`Self::from_card`] — which never attempts the extended-card upgrade,
    /// matching its documented "no discovery call is ever made" contract),
    /// later calls return the cached value without any network access.
    pub async fn get_agent_card(&self) -> Result<AgentCard> {
        if let Some(card) = self.cached_agent_card().await {
            return Ok(card);
        }
        let base = self.discovery_base.trim_end_matches('/');
        let primary = format!("{base}{WELL_KNOWN_AGENT_CARD_PATH}");
        let mut card = match self.get_json::<AgentCard>(&primary).await {
            Ok(card) => card,
            Err(primary_err) => {
                let fallback = format!("{base}{WELL_KNOWN_AGENT_JSON_PATH}");
                self.get_json::<AgentCard>(&fallback)
                    .await
                    .map_err(|fallback_err| {
                        Error::service(format!(
                            "A2A agent card discovery failed at '{primary}' ({primary_err}) \
                             and '{fallback}' ({fallback_err})"
                        ))
                    })?
            }
        };
        *self.rpc_url.write().await = card.url.clone();
        if card.supports_authenticated_extended_card {
            match self.get_extended_card().await {
                Ok(extended) => card = extended,
                Err(e) => {
                    tracing::debug!(
                        error = %e,
                        "A2A: authenticated extended card fetch failed; using the base card"
                    );
                }
            }
        }
        *self.card.write().await = Some(card.clone());
        Ok(card)
    }

    /// `agent/getAuthenticatedExtendedCard`: fetch the agent's extended
    /// [`AgentCard`] directly, using whatever auth headers this client is
    /// configured with (see [`Self::with_header`] / [`Self::with_bearer_token`]).
    ///
    /// Low-level: does not check `supportsAuthenticatedExtendedCard`, does
    /// not touch the cache, and propagates failure — [`Self::get_agent_card`]
    /// wraps this with that capability check and a graceful fallback for its
    /// own automatic upgrade. Call this directly when you want the extended
    /// card specifically and want to know if that actually succeeded.
    pub async fn get_extended_card(&self) -> Result<AgentCard> {
        let raw = self
            .call("agent/getAuthenticatedExtendedCard", json!({}))
            .await?;
        serde_json::from_value(raw)
            .map_err(|e| Error::Serialization(format!("invalid A2A AgentCard: {e}")))
    }

    async fn get_json<T: DeserializeOwned>(&self, url: &str) -> Result<T> {
        let resp = self
            .http
            .get(url)
            .headers(self.headers.clone())
            .send()
            .await
            .map_err(|e| Error::service(format!("GET {url} failed: {e}")))?;
        let status = resp.status();
        if !status.is_success() {
            return Err(Error::service(format!("GET {url} returned {status}")));
        }
        resp.json::<T>()
            .await
            .map_err(|e| Error::service(format!("invalid JSON from {url}: {e}")))
    }

    /// `message/send`: send a message and get back either an immediate
    /// [`Message`](crate::types::Message) reply, or a created/continued
    /// [`Task`].
    pub async fn send_message(&self, params: MessageSendParams) -> Result<SendMessageResult> {
        let raw = self
            .call("message/send", serde_json::to_value(params)?)
            .await?;
        protocol::parse_send_message_result(&raw)
    }

    /// `tasks/get`: fetch a task's current state.
    pub async fn get_task(&self, task_id: &str) -> Result<Task> {
        let raw = self.call("tasks/get", json!({ "id": task_id })).await?;
        serde_json::from_value(raw)
            .map_err(|e| Error::Serialization(format!("invalid A2A Task: {e}")))
    }

    /// `tasks/cancel`: request cancellation of a task.
    pub async fn cancel_task(&self, task_id: &str) -> Result<Task> {
        let raw = self.call("tasks/cancel", json!({ "id": task_id })).await?;
        serde_json::from_value(raw)
            .map_err(|e| Error::Serialization(format!("invalid A2A Task: {e}")))
    }

    /// `tasks/pushNotificationConfig/set`: register a webhook the server
    /// should call as `task_id` progresses. Returns the config the server
    /// actually stored (it may assign a `PushNotificationConfig::id` if one
    /// wasn't given).
    pub async fn set_push_notification_config(
        &self,
        task_id: &str,
        config: PushNotificationConfig,
    ) -> Result<TaskPushNotificationConfig> {
        let params = TaskPushNotificationConfig {
            task_id: task_id.to_string(),
            push_notification_config: config,
        };
        let raw = self
            .call(
                "tasks/pushNotificationConfig/set",
                serde_json::to_value(&params)?,
            )
            .await?;
        serde_json::from_value(raw).map_err(|e| {
            Error::Serialization(format!("invalid A2A TaskPushNotificationConfig: {e}"))
        })
    }

    /// `tasks/pushNotificationConfig/get`: fetch the push notification
    /// config currently registered for `task_id`.
    ///
    /// Note the params shape genuinely differs from
    /// [`Self::set_push_notification_config`]'s: the A2A 0.3.0 spec sends
    /// the task id under `id` here (`GetTaskPushNotificationConfigParams`),
    /// not `taskId` — a real wire-level inconsistency in the spec/SDK, not a
    /// mistake in this port.
    pub async fn get_push_notification_config(
        &self,
        task_id: &str,
    ) -> Result<TaskPushNotificationConfig> {
        let raw = self
            .call("tasks/pushNotificationConfig/get", json!({ "id": task_id }))
            .await?;
        serde_json::from_value(raw).map_err(|e| {
            Error::Serialization(format!("invalid A2A TaskPushNotificationConfig: {e}"))
        })
    }

    /// `message/stream`: like [`Self::send_message`], but returns a stream of
    /// [`MessageStreamEvent`]s — an immediate
    /// [`Message`](crate::types::Message), or a sequence of `Task`
    /// status/artifact updates — as the server emits them over
    /// Server-Sent-Events.
    ///
    /// This is lower-level than [`Self::send_message`]:
    /// `A2AAgent::run` does not use it (see the crate docs for why),
    /// but it's available for callers that want incremental updates.
    pub async fn send_message_stream(&self, params: MessageSendParams) -> Result<A2AEventStream> {
        self.stream_request("message/stream", serde_json::to_value(params)?)
            .await
    }

    /// `tasks/resubscribe`: reconnect to an existing task's event stream
    /// (e.g. after a dropped connection), yielding the exact same
    /// [`MessageStreamEvent`] sequence shape as [`Self::send_message_stream`]
    /// — per the A2A spec, both requests share one response type on the
    /// wire.
    pub async fn resubscribe(&self, task_id: &str) -> Result<A2AEventStream> {
        self.stream_request("tasks/resubscribe", json!({ "id": task_id }))
            .await
    }

    /// POST `method`/`params` requesting `text/event-stream` and return the
    /// resulting SSE-framed event stream. Shared by
    /// [`Self::send_message_stream`] and [`Self::resubscribe`], which differ
    /// only in the method name and params shape — both respond with the same
    /// `Message | Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent`
    /// union over SSE.
    async fn stream_request(&self, method: &str, params: Value) -> Result<A2AEventStream> {
        let url = self.rpc_url().await;
        let id = uuid::Uuid::new_v4().to_string();
        let body = protocol::build_request(&id, method, params);
        let resp = self
            .http
            .post(&url)
            .headers(self.headers.clone())
            .header(ACCEPT, "text/event-stream")
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::service(format!("A2A request to {url} failed: {e}")))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(Error::service(format!("A2A HTTP {status}: {text}")));
        }
        Ok(parse_sse_stream(resp).boxed())
    }

    async fn call(&self, method: &str, params: Value) -> Result<Value> {
        let url = self.rpc_url().await;
        let id = uuid::Uuid::new_v4().to_string();
        let body = protocol::build_request(&id, method, params);
        let resp = self
            .http
            .post(&url)
            .headers(self.headers.clone())
            .header(ACCEPT, "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::service(format!("A2A request to {url} failed: {e}")))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(Error::service(format!("A2A HTTP {status}: {text}")));
        }
        let value: Value = resp
            .json()
            .await
            .map_err(|e| Error::service(format!("invalid A2A JSON response: {e}")))?;
        protocol::extract_result(&value)
    }
}

type ByteStream = Pin<Box<dyn Stream<Item = reqwest::Result<bytes::Bytes>> + Send>>;

/// State carried across `unfold` iterations while parsing the `message/stream`
/// SSE response, mirroring `agent_framework_openai`'s chat-completions SSE
/// parser.
struct SseState {
    byte_stream: ByteStream,
    buffer: String,
    queued: VecDeque<Result<MessageStreamEvent>>,
    done: bool,
}

/// Turn an SSE HTTP response into a stream of [`MessageStreamEvent`]s.
fn parse_sse_stream(
    resp: reqwest::Response,
) -> impl Stream<Item = Result<MessageStreamEvent>> + Send {
    let byte_stream: ByteStream = Box::pin(resp.bytes_stream());
    futures::stream::unfold(
        SseState {
            byte_stream,
            buffer: String::new(),
            queued: VecDeque::new(),
            done: false,
        },
        |mut state| async move {
            loop {
                if let Some(item) = state.queued.pop_front() {
                    return Some((item, state));
                }
                if state.done {
                    return None;
                }
                match state.byte_stream.next().await {
                    Some(Ok(bytes)) => {
                        state.buffer.push_str(&String::from_utf8_lossy(&bytes));
                        let events = protocol::drain_sse_events(&mut state.buffer);
                        state.queued.extend(events);
                    }
                    Some(Err(e)) => {
                        state.done = true;
                        return Some((
                            Err(Error::service(format!("A2A stream error: {e}"))),
                            state,
                        ));
                    }
                    None => {
                        // A trailing, unterminated buffered event (no closing
                        // "\n\n") is deliberately dropped: without it we
                        // cannot tell the frame was fully sent. `state` (and
                        // its `done` flag) is dropped here too, since `None`
                        // permanently ends the `unfold` stream.
                        return None;
                    }
                }
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Message, MessageRole, Part, SendMessageResult, TextPart};

    fn sample_card() -> AgentCard {
        AgentCard {
            name: "Test Agent".into(),
            url: "https://agent.example.com/rpc".into(),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn from_url_uses_url_as_rpc_endpoint_and_has_no_cached_card() {
        let client = A2AClient::from_url("https://agent.example.com/rpc");
        assert_eq!(client.rpc_url().await, "https://agent.example.com/rpc");
        assert!(client.cached_agent_card().await.is_none());
    }

    #[tokio::test]
    async fn from_card_caches_card_and_uses_its_url() {
        let card = sample_card();
        let client = A2AClient::from_card(card.clone());
        assert_eq!(client.rpc_url().await, card.url);
        assert_eq!(client.cached_agent_card().await, Some(card));
    }

    #[tokio::test]
    async fn get_agent_card_returns_cached_card_without_any_network_access() {
        let card = sample_card();
        let client = A2AClient::from_card(card.clone());
        // Cached (from `from_card`), so this must not attempt any I/O.
        let fetched = client.get_agent_card().await.unwrap();
        assert_eq!(fetched, card);
    }

    #[test]
    fn with_header_rejects_invalid_header_name() {
        let err = A2AClient::from_url("https://example.com")
            .with_header("bad header", "value")
            .unwrap_err();
        assert!(matches!(err, Error::Configuration(_)));
    }

    #[test]
    fn with_header_rejects_invalid_header_value() {
        let err = A2AClient::from_url("https://example.com")
            .with_header("X-Test", "bad\nvalue")
            .unwrap_err();
        assert!(matches!(err, Error::Configuration(_)));
    }

    #[test]
    fn with_bearer_token_sets_authorization_header() {
        let client = A2AClient::from_url("https://example.com")
            .with_bearer_token("secret123")
            .unwrap();
        assert_eq!(
            client.headers.get("authorization").unwrap(),
            "Bearer secret123"
        );
    }

    #[test]
    fn with_timeout_succeeds() {
        let client = A2AClient::from_url("https://example.com")
            .with_timeout(Duration::from_secs(5))
            .unwrap();
        assert_eq!(client.discovery_base, "https://example.com");
    }

    #[test]
    fn builder_chain_composes() {
        let client = A2AClient::from_url("https://example.com")
            .with_bearer_token("tok")
            .unwrap()
            .with_header("X-Trace", "abc")
            .unwrap();
        assert_eq!(client.headers.len(), 2);
    }

    #[tokio::test]
    async fn drain_sse_events_feeds_message_stream_event_type_end_to_end() {
        // Exercises the same MessageStreamEvent type send_message_stream
        // yields, via the pure SSE parser (no network).
        let message = Message {
            role: MessageRole::Agent,
            parts: vec![Part::Text(TextPart {
                text: "hi".into(),
                metadata: None,
            })],
            message_id: "m1".into(),
            task_id: None,
            context_id: None,
            metadata: None,
        };
        let mut buf = format!(
            "data: {}\n\n",
            json!({
                "jsonrpc": "2.0",
                "id": "1",
                "result": serde_json::to_value(SendMessageResult::Message(message)).unwrap(),
            })
        );
        let events = protocol::drain_sse_events(&mut buf);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0].as_ref().unwrap(),
            MessageStreamEvent::Message(_)
        ));
    }

    #[test]
    fn resubscribe_reuses_the_same_sse_event_parsing_as_message_stream() {
        // `tasks/resubscribe` and `message/stream` share one response shape
        // on the wire (a `Message | Task | TaskStatusUpdateEvent |
        // TaskArtifactUpdateEvent` union over SSE) -- `A2AClient::resubscribe`
        // is implemented via the exact same `stream_request` helper (and
        // thus the exact same `parse_sse_stream` / `drain_sse_events` /
        // `protocol::parse_stream_event` path) as `send_message_stream`, so
        // this exercises a status-update event the same way a resumed task
        // stream would deliver one, with no special-casing anywhere.
        let mut buf = format!(
            "data: {}\n\n",
            json!({
                "jsonrpc": "2.0",
                "id": "1",
                "result": {
                    "taskId": "task-1",
                    "contextId": "ctx-1",
                    "status": {"state": "working"},
                    "final": false,
                },
            })
        );
        let events = protocol::drain_sse_events(&mut buf);
        assert_eq!(events.len(), 1);
        match events[0].as_ref().unwrap() {
            MessageStreamEvent::StatusUpdate(update) => {
                assert_eq!(update.task_id, "task-1");
                assert!(!update.is_final);
            }
            other => panic!("expected StatusUpdate, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn from_card_never_attempts_the_extended_card_upgrade() {
        // `from_card` documents "no discovery call is ever made"; the
        // extended-card auto-upgrade lives inside `get_agent_card`'s
        // discovery path, so a client built from an already-known card must
        // short-circuit before ever reaching it, even when that card claims
        // `supportsAuthenticatedExtendedCard`. No network access happens in
        // this test at all -- if the upgrade were mistakenly attempted, this
        // would hang/fail trying to reach a real host.
        let mut card = sample_card();
        card.supports_authenticated_extended_card = true;
        let client = A2AClient::from_card(card.clone());
        let fetched = client.get_agent_card().await.unwrap();
        assert_eq!(fetched, card);
    }
}
