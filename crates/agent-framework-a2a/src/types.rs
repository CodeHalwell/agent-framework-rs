//! A2A protocol domain types: [`AgentCard`], [`Message`]/[`Part`], and
//! [`Task`]/[`Artifact`], serialized exactly as the A2A JSON-RPC wire format
//! (camelCase field names, `kind` discriminators where the spec uses one).
//!
//! Deserialization is deliberately tolerant of fields this crate doesn't
//! model: no type here uses `deny_unknown_fields`, so a server that adds new
//! properties in a future spec revision (or a proprietary extension) won't
//! break parsing — the extra fields are just ignored.

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------------
// AgentCard
// ---------------------------------------------------------------------

/// An A2A agent's self-description, normally fetched from
/// `/.well-known/agent-card.json` (see
/// [`A2AClient::get_agent_card`](crate::client::A2AClient::get_agent_card)).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentCard {
    /// The A2A protocol version the server implements, e.g. `"0.3.0"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protocol_version: Option<String>,
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// The JSON-RPC endpoint URL: every [`A2AClient`](crate::client::A2AClient)
    /// request is POSTed here.
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preferred_transport: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<AgentProvider>,
    #[serde(default)]
    pub version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub documentation_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub icon_url: Option<String>,
    #[serde(default)]
    pub capabilities: AgentCapabilities,
    #[serde(default)]
    pub default_input_modes: Vec<String>,
    #[serde(default)]
    pub default_output_modes: Vec<String>,
    #[serde(default)]
    pub skills: Vec<AgentSkill>,
    #[serde(default)]
    pub supports_authenticated_extended_card: bool,
    /// Security scheme definitions, kept as raw JSON: the full A2A security
    /// model (OAuth2/OpenIdConnect/apiKey/http scheme unions) is out of scope
    /// for this client, which does not automate credential negotiation —
    /// see [`A2AClient::with_header`](crate::client::A2AClient::with_header).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub security_schemes: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub security: Option<Value>,
}

impl AgentCard {
    /// Build a minimal card that just points at `url`, with every other
    /// field defaulted. Used when no real discovery document is available;
    /// mirrors the Python reference's `minimal_agent_card` fallback.
    pub fn minimal(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            ..Default::default()
        }
    }
}

/// The organization behind an [`AgentCard`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentProvider {
    #[serde(default)]
    pub organization: String,
    #[serde(default)]
    pub url: String,
}

/// What an A2A agent supports, beyond the baseline `message/send`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentCapabilities {
    #[serde(default)]
    pub streaming: bool,
    #[serde(default)]
    pub push_notifications: bool,
    #[serde(default)]
    pub state_transition_history: bool,
    #[serde(default)]
    pub extensions: Vec<Value>,
}

/// One capability an [`AgentCard`] advertises.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentSkill {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub examples: Vec<String>,
    #[serde(default)]
    pub input_modes: Vec<String>,
    #[serde(default)]
    pub output_modes: Vec<String>,
}

// ---------------------------------------------------------------------
// Message & Part
// ---------------------------------------------------------------------

/// Who sent a [`Message`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Agent,
}

/// One turn of an A2A conversation: a list of [`Part`]s plus routing
/// metadata (`taskId`/`contextId`) used to continue a conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Message {
    pub role: MessageRole,
    pub parts: Vec<Part>,
    pub message_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

impl Message {
    /// Build a `user`-role message with the given parts and a fresh random id.
    pub fn user(parts: Vec<Part>) -> Self {
        Self {
            role: MessageRole::User,
            parts,
            message_id: uuid::Uuid::new_v4().to_string(),
            task_id: None,
            context_id: None,
            metadata: None,
        }
    }
}

/// A single piece of a [`Message`] or [`Artifact`]: text, a file (inline
/// bytes or a URI), or arbitrary structured data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Part {
    Text(TextPart),
    File(FilePart),
    Data(DataPart),
}

/// A plain-text [`Part`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TextPart {
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// A file [`Part`]: either inline base64 bytes or a URI reference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilePart {
    pub file: FileData,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// A [`FilePart`]'s payload.
///
/// Untagged: disambiguated structurally (`bytes` vs. `uri` is a required
/// field unique to each variant), matching the A2A JSON schema, which does
/// not add its own discriminator here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FileData {
    Bytes(FileWithBytes),
    Uri(FileWithUri),
}

/// Inline file content, base64-encoded.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileWithBytes {
    /// Base64-encoded file content.
    pub bytes: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// A file referenced by URI rather than inlined.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileWithUri {
    pub uri: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// A structured-data [`Part`] (arbitrary JSON, not text or a file).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DataPart {
    pub data: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

// ---------------------------------------------------------------------
// Task & Artifact
// ---------------------------------------------------------------------

/// The lifecycle state of a [`Task`].
///
/// `Unknown` is both a real spec value (`"unknown"`) and this enum's fallback
/// for any state string a future spec revision might add (`#[serde(other)]`),
/// so parsing a [`Task`] never fails just because the server reports a state
/// this crate predates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TaskState {
    Submitted,
    Working,
    InputRequired,
    Completed,
    Canceled,
    Failed,
    Rejected,
    AuthRequired,
    #[serde(other)]
    Unknown,
}

impl TaskState {
    /// Whether this state ends the task's lifecycle (no further updates).
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            TaskState::Completed | TaskState::Canceled | TaskState::Failed | TaskState::Rejected
        )
    }
}

/// A [`Task`]'s current status: state, optional accompanying message (e.g. a
/// clarifying question when `state` is [`TaskState::InputRequired`]), and
/// timestamp.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskStatus {
    pub state: TaskState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<Message>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
}

/// A unit of work an A2A agent is carrying out, identified by `id` within a
/// `contextId` (conversation).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Task {
    pub id: String,
    pub context_id: String,
    pub status: TaskStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifacts: Option<Vec<Artifact>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub history: Option<Vec<Message>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// A named output produced by a [`Task`] (e.g. a generated document).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Artifact {
    pub artifact_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parts: Vec<Part>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

// ---------------------------------------------------------------------
// message/stream update events
// ---------------------------------------------------------------------

/// A `message/stream` event carrying a [`Task`]'s new [`TaskStatus`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskStatusUpdateEvent {
    pub task_id: String,
    pub context_id: String,
    pub status: TaskStatus,
    /// Whether this is the last event for the stream. The wire field is
    /// named `final`, a reserved-ish word best avoided as a Rust identifier.
    #[serde(rename = "final", default)]
    pub is_final: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// A `message/stream` event carrying a newly produced/updated [`Artifact`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskArtifactUpdateEvent {
    pub task_id: String,
    pub context_id: String,
    pub artifact: Artifact,
    #[serde(default)]
    pub append: bool,
    #[serde(default)]
    pub last_chunk: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

// ---------------------------------------------------------------------
// message/send and message/stream results
// ---------------------------------------------------------------------

/// The result of `message/send`: the server either answers immediately with
/// a [`Message`], or creates/continues a [`Task`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum SendMessageResult {
    Message(Message),
    Task(Task),
}

/// One event from `message/stream`: everything [`SendMessageResult`] can be,
/// plus incremental [`Task`] status/artifact updates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum MessageStreamEvent {
    Message(Message),
    Task(Task),
    StatusUpdate(TaskStatusUpdateEvent),
    ArtifactUpdate(TaskArtifactUpdateEvent),
}

// ---------------------------------------------------------------------
// message/send, message/stream params
// ---------------------------------------------------------------------

/// Parameters for `message/send` / `message/stream`.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageSendParams {
    pub message: Message,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub configuration: Option<MessageSendConfiguration>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

impl MessageSendParams {
    /// Params carrying just a message, no special configuration.
    pub fn new(message: Message) -> Self {
        Self {
            message,
            configuration: None,
            metadata: None,
        }
    }
}

/// Optional hints for how the server should handle a `message/send` /
/// `message/stream` call.
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageSendConfiguration {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accepted_output_modes: Option<Vec<String>>,
    /// Ask the server to hold the response until the task reaches a terminal
    /// or input-required state, rather than returning immediately. Advisory:
    /// not every server honors it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocking: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub history_length: Option<u32>,
}

// ---------------------------------------------------------------------
// Push notification config (`tasks/pushNotificationConfig/set` / `/get`)
// ---------------------------------------------------------------------

/// How the server should authenticate itself when it calls back a
/// [`PushNotificationConfig::url`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PushNotificationAuthenticationInfo {
    pub schemes: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credentials: Option<String>,
}

/// A webhook the server should call as a [`Task`] progresses, set via
/// `tasks/pushNotificationConfig/set`
/// ([`A2AClient::set_push_notification_config`](crate::client::A2AClient::set_push_notification_config)).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PushNotificationConfig {
    /// Identifies this config among possibly several registered for the
    /// same task (A2A 0.3.0+). Optional: a server may assign one if omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authentication: Option<PushNotificationAuthenticationInfo>,
}

impl PushNotificationConfig {
    /// Build a config with just a callback URL.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            id: None,
            url: url.into(),
            token: None,
            authentication: None,
        }
    }

    /// Set a shared-secret token the server should include on its callback.
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Set the authentication schemes/credentials the server should use.
    pub fn with_authentication(
        mut self,
        authentication: PushNotificationAuthenticationInfo,
    ) -> Self {
        self.authentication = Some(authentication);
        self
    }
}

/// The `taskId` + [`PushNotificationConfig`] pair used as both the params of
/// `tasks/pushNotificationConfig/set` and the result of both `set` and `get`.
///
/// Note: the `get` request's *params* use a different shape
/// (`GetTaskPushNotificationConfigParams`, with the task id under `id` —
/// see [`A2AClient::get_push_notification_config`](crate::client::A2AClient::get_push_notification_config)),
/// an actual wire-level inconsistency in the A2A 0.3.0 spec/SDK, not a typo
/// here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskPushNotificationConfig {
    pub task_id: String,
    pub push_notification_config: PushNotificationConfig,
}

#[cfg(test)]
mod tests {
    use super::*;

    const FULL_CARD_FIXTURE: &str = r#"{
        "protocolVersion": "0.3.0",
        "name": "Weather Agent",
        "description": "Provides weather forecasts and alerts.",
        "url": "https://weather.example.com/a2a/v1",
        "preferredTransport": "JSONRPC",
        "provider": {
            "organization": "Example Corp",
            "url": "https://example.com"
        },
        "version": "1.2.0",
        "documentationUrl": "https://weather.example.com/docs",
        "iconUrl": "https://weather.example.com/icon.png",
        "capabilities": {
            "streaming": true,
            "pushNotifications": false,
            "stateTransitionHistory": true,
            "extensions": [],
            "someFutureCapability": true
        },
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain", "application/json"],
        "skills": [
            {
                "id": "get-forecast",
                "name": "Get Forecast",
                "description": "Returns a multi-day forecast for a location.",
                "tags": ["weather", "forecast"],
                "examples": ["What's the weather in Seattle?"],
                "inputModes": ["text/plain"],
                "outputModes": ["text/plain"]
            }
        ],
        "supportsAuthenticatedExtendedCard": true,
        "someFutureTopLevelField": {
            "nested": true
        }
    }"#;

    #[test]
    fn agent_card_parses_realistic_fixture_and_ignores_unknown_fields() {
        let card: AgentCard = serde_json::from_str(FULL_CARD_FIXTURE).unwrap();
        assert_eq!(card.name, "Weather Agent");
        assert_eq!(card.url, "https://weather.example.com/a2a/v1");
        assert_eq!(card.protocol_version.as_deref(), Some("0.3.0"));
        assert_eq!(card.version, "1.2.0");
        assert!(card.capabilities.streaming);
        assert!(!card.capabilities.push_notifications);
        assert!(card.capabilities.state_transition_history);
        assert_eq!(card.default_input_modes, vec!["text/plain"]);
        assert_eq!(
            card.default_output_modes,
            vec!["text/plain", "application/json"]
        );
        assert_eq!(card.skills.len(), 1);
        assert_eq!(card.skills[0].id, "get-forecast");
        assert_eq!(card.skills[0].tags, vec!["weather", "forecast"]);
        assert!(card.supports_authenticated_extended_card);
        assert_eq!(
            card.provider.as_ref().map(|p| p.organization.as_str()),
            Some("Example Corp")
        );
    }

    #[test]
    fn agent_card_tolerates_missing_optional_fields() {
        let minimal = r#"{"name": "Minimal", "url": "https://x.example/rpc"}"#;
        let card: AgentCard = serde_json::from_str(minimal).unwrap();
        assert_eq!(card.name, "Minimal");
        assert_eq!(card.description, "");
        assert_eq!(card.version, "");
        assert!(card.skills.is_empty());
        assert!(card.default_input_modes.is_empty());
        assert!(!card.capabilities.streaming);
        assert!(card.protocol_version.is_none());
    }

    #[test]
    fn agent_card_minimal_constructor_sets_only_url() {
        let card = AgentCard::minimal("https://example.com/rpc");
        assert_eq!(card.url, "https://example.com/rpc");
        assert_eq!(card.name, "");
    }

    #[test]
    fn message_role_serializes_lowercase() {
        assert_eq!(serde_json::to_value(MessageRole::User).unwrap(), "user");
        assert_eq!(serde_json::to_value(MessageRole::Agent).unwrap(), "agent");
        let role: MessageRole = serde_json::from_str("\"agent\"").unwrap();
        assert_eq!(role, MessageRole::Agent);
    }

    #[test]
    fn part_text_round_trips_with_kind_tag() {
        let part = Part::Text(TextPart {
            text: "hello".into(),
            metadata: None,
        });
        let value = serde_json::to_value(&part).unwrap();
        assert_eq!(value["kind"], "text");
        assert_eq!(value["text"], "hello");
        let parsed: Part = serde_json::from_value(value).unwrap();
        assert_eq!(parsed, part);
    }

    #[test]
    fn part_file_with_uri_round_trips() {
        let json = serde_json::json!({
            "kind": "file",
            "file": { "uri": "https://example.com/report.pdf", "mimeType": "application/pdf" }
        });
        let part: Part = serde_json::from_value(json).unwrap();
        match &part {
            Part::File(f) => match &f.file {
                FileData::Uri(u) => {
                    assert_eq!(u.uri, "https://example.com/report.pdf");
                    assert_eq!(u.mime_type.as_deref(), Some("application/pdf"));
                }
                FileData::Bytes(_) => panic!("expected a URI file"),
            },
            other => panic!("expected a file part, got {other:?}"),
        }
    }

    #[test]
    fn part_file_with_bytes_round_trips() {
        let json = serde_json::json!({
            "kind": "file",
            "file": { "bytes": "aGVsbG8=", "mimeType": "text/plain", "name": "hello.txt" }
        });
        let part: Part = serde_json::from_value(json).unwrap();
        match &part {
            Part::File(f) => match &f.file {
                FileData::Bytes(b) => {
                    assert_eq!(b.bytes, "aGVsbG8=");
                    assert_eq!(b.name.as_deref(), Some("hello.txt"));
                }
                FileData::Uri(_) => panic!("expected a bytes file"),
            },
            other => panic!("expected a file part, got {other:?}"),
        }
    }

    #[test]
    fn part_data_round_trips() {
        let json = serde_json::json!({ "kind": "data", "data": { "temp_f": 72 } });
        let part: Part = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(
            part,
            Part::Data(DataPart {
                data: serde_json::json!({"temp_f": 72}),
                metadata: None
            })
        );
        assert_eq!(serde_json::to_value(&part).unwrap()["kind"], "data");
    }

    #[test]
    fn task_state_parses_all_spec_values() {
        let cases = [
            ("\"submitted\"", TaskState::Submitted),
            ("\"working\"", TaskState::Working),
            ("\"input-required\"", TaskState::InputRequired),
            ("\"completed\"", TaskState::Completed),
            ("\"canceled\"", TaskState::Canceled),
            ("\"failed\"", TaskState::Failed),
            ("\"rejected\"", TaskState::Rejected),
            ("\"auth-required\"", TaskState::AuthRequired),
            ("\"unknown\"", TaskState::Unknown),
        ];
        for (json, expected) in cases {
            let parsed: TaskState = serde_json::from_str(json).unwrap();
            assert_eq!(parsed, expected, "parsing {json}");
        }
    }

    #[test]
    fn task_state_unrecognized_value_falls_back_to_unknown() {
        let parsed: TaskState = serde_json::from_str("\"some-future-state\"").unwrap();
        assert_eq!(parsed, TaskState::Unknown);
    }

    #[test]
    fn task_state_is_terminal() {
        assert!(TaskState::Completed.is_terminal());
        assert!(TaskState::Failed.is_terminal());
        assert!(TaskState::Canceled.is_terminal());
        assert!(TaskState::Rejected.is_terminal());
        assert!(!TaskState::Working.is_terminal());
        assert!(!TaskState::InputRequired.is_terminal());
        assert!(!TaskState::Submitted.is_terminal());
        assert!(!TaskState::AuthRequired.is_terminal());
        assert!(!TaskState::Unknown.is_terminal());
    }

    #[test]
    fn task_deserializes_with_artifacts_and_history() {
        let json = serde_json::json!({
            "id": "task-1",
            "contextId": "ctx-1",
            "status": { "state": "completed" },
            "artifacts": [
                { "artifactId": "art-1", "parts": [{"kind": "text", "text": "done"}] }
            ],
            "history": [
                { "role": "user", "parts": [{"kind": "text", "text": "hi"}], "messageId": "m1" }
            ]
        });
        let task: Task = serde_json::from_value(json).unwrap();
        assert_eq!(task.id, "task-1");
        assert_eq!(task.context_id, "ctx-1");
        assert_eq!(task.status.state, TaskState::Completed);
        assert_eq!(task.artifacts.as_ref().unwrap().len(), 1);
        assert_eq!(task.history.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn task_status_update_event_maps_final_keyword_field() {
        let json = serde_json::json!({
            "taskId": "task-1",
            "contextId": "ctx-1",
            "status": { "state": "working" },
            "final": true
        });
        let event: TaskStatusUpdateEvent = serde_json::from_value(json).unwrap();
        assert!(event.is_final);
        assert_eq!(serde_json::to_value(&event).unwrap()["final"], true);
    }

    #[test]
    fn message_send_params_serializes_without_optional_fields() {
        let message = Message::user(vec![Part::Text(TextPart {
            text: "hi".into(),
            metadata: None,
        })]);
        let params = MessageSendParams::new(message);
        let value = serde_json::to_value(&params).unwrap();
        assert!(value.get("configuration").is_none());
        assert!(value.get("metadata").is_none());
        assert_eq!(value["message"]["role"], "user");
    }

    // -- Push notification config -----------------------------------------

    #[test]
    fn push_notification_config_builder_composes() {
        let config = PushNotificationConfig::new("https://example.com/hook")
            .with_token("secret")
            .with_authentication(PushNotificationAuthenticationInfo {
                schemes: vec!["Bearer".into()],
                credentials: Some("abc".into()),
            });
        assert_eq!(config.url, "https://example.com/hook");
        assert_eq!(config.token.as_deref(), Some("secret"));
        assert_eq!(
            config.authentication.as_ref().unwrap().schemes,
            vec!["Bearer".to_string()]
        );
    }

    #[test]
    fn push_notification_config_serializes_camel_case_and_omits_unset_fields() {
        let config = PushNotificationConfig::new("https://example.com/hook");
        let value = serde_json::to_value(&config).unwrap();
        assert_eq!(value["url"], "https://example.com/hook");
        assert!(value.get("id").is_none());
        assert!(value.get("token").is_none());
        assert!(value.get("authentication").is_none());
    }

    #[test]
    fn task_push_notification_config_round_trips_with_camel_case_task_id() {
        let json = serde_json::json!({
            "taskId": "task-1",
            "pushNotificationConfig": {
                "url": "https://example.com/hook",
                "authentication": {"schemes": ["Bearer"], "credentials": "tok"},
            },
        });
        let parsed: TaskPushNotificationConfig = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(parsed.task_id, "task-1");
        assert_eq!(
            parsed.push_notification_config.url,
            "https://example.com/hook"
        );
        assert_eq!(
            parsed
                .push_notification_config
                .authentication
                .as_ref()
                .unwrap()
                .credentials
                .as_deref(),
            Some("tok")
        );
        assert_eq!(serde_json::to_value(&parsed).unwrap(), json);
    }

    #[test]
    fn task_push_notification_config_deserializes_with_config_id() {
        let json = serde_json::json!({
            "taskId": "task-1",
            "pushNotificationConfig": {
                "id": "cfg-1",
                "url": "https://example.com/hook",
            },
        });
        let parsed: TaskPushNotificationConfig = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.push_notification_config.id.as_deref(), Some("cfg-1"));
    }
}
