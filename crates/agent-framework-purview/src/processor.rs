//! [`ContentProcessor`]: maps [`ChatMessage`]s to `processContent` requests,
//! resolves the acting user id, and evaluates the resulting verdicts.
//!
//! A scoped-down port of Python's `ScopedContentProcessor` — see
//! [`crate::client`]'s module docs for exactly what's cut (the protection-
//! scopes precheck, caching, and background content-activity logging) and
//! why. What *is* ported faithfully: per-message request construction
//! (the internal `build_request`) and the GUID-based user-id resolution algorithm
//! (mirrors `ScopedContentProcessor._map_messages`'s
//! `additional_properties["user_id"]` / `author_name` scan, minus the
//! bearer-token-JWT fallback — see the crate docs).

use agent_framework_core::error::{Error, Result};
use agent_framework_core::types::ChatMessage;

use crate::client::PurviewClient;
use crate::models::{
    Activity, ActivityMetadata, ContentToProcess, DeviceMetadata, IntegratedAppMetadata,
    ProcessContentRequest, ProcessConversationMetadata, ProtectedAppMetadata,
};
use crate::settings::PurviewSettings;

/// Validate a string as a GUID/UUID, mirroring Python's `_is_valid_guid`
/// (`uuid.UUID(value)` succeeding).
fn is_valid_guid(value: &str) -> bool {
    uuid::Uuid::parse_str(value).is_ok()
}

/// Resolve the acting user id for a batch of messages, mirroring
/// `ScopedContentProcessor._map_messages`'s resolution order (minus the
/// bearer-token-JWT fallback this port doesn't perform — see the crate
/// docs):
///
/// 1. The first message whose `additional_properties["user_id"]` is a valid
///    GUID wins outright.
/// 2. Otherwise, the first message whose `author_name` is a valid GUID is
///    remembered as a fallback candidate (scanning continues, in case a
///    later message has an explicit `user_id`).
/// 3. If neither produced a value, `provided` is used if it is itself a
///    valid GUID.
/// 4. Otherwise `None` — callers must treat this as "cannot evaluate; do not
///    block" (see [`ContentProcessor::evaluate`]), matching Python's
///    fail-open behavior when no resolvable user id exists.
pub fn resolve_user_id(messages: &[ChatMessage], provided: Option<&str>) -> Option<String> {
    let mut author_name_fallback: Option<String> = None;
    for message in messages {
        if let Some(user_id) = message
            .additional_properties
            .get("user_id")
            .and_then(serde_json::Value::as_str)
        {
            if is_valid_guid(user_id) {
                return Some(user_id.to_string());
            }
        }
        if author_name_fallback.is_none() {
            if let Some(name) = &message.author_name {
                if is_valid_guid(name) {
                    author_name_fallback = Some(name.clone());
                }
            }
        }
    }
    author_name_fallback.or_else(|| provided.filter(|p| is_valid_guid(p)).map(str::to_string))
}

/// Build one `processContent` request for a single message. Mirrors the body
/// of `ScopedContentProcessor._map_messages`'s per-message loop (device
/// metadata is always `"Unknown"`/`"Unknown"`, matching Python's hardcoded
/// values there).
fn build_request(
    message: &ChatMessage,
    user_id: &str,
    tenant_id: &str,
    app_name: &str,
    app_location: &crate::models::PolicyLocation,
) -> ProcessContentRequest {
    let message_id = message
        .message_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let entry = ProcessConversationMetadata::new(
        message_id.clone(),
        message.text(),
        format!("Agent Framework Message {message_id}"),
    );
    let content_to_process = ContentToProcess {
        content_entries: vec![entry],
        // Both the prompt (pre) and response (post) checks use `UploadText`
        // — see the crate docs' "A curious fidelity note" section for why
        // this mirrors Python exactly rather than using `DownloadText` for
        // the response direction.
        activity_metadata: ActivityMetadata {
            activity: Activity::UploadText,
        },
        device_metadata: DeviceMetadata::default(),
        integrated_app_metadata: IntegratedAppMetadata {
            name: app_name.to_string(),
            version: "1.0".to_string(),
        },
        protected_app_metadata: ProtectedAppMetadata {
            name: app_name.to_string(),
            version: "1.0".to_string(),
            application_location: app_location.clone(),
        },
    };
    ProcessContentRequest {
        content_to_process,
        user_id: user_id.to_string(),
        tenant_id: tenant_id.to_string(),
        correlation_id: Some(uuid::Uuid::new_v4().to_string()),
    }
}

/// Orchestrates `processContent` evaluation over a batch of messages. See
/// the module docs for how this differs in scope from Python's
/// `ScopedContentProcessor`.
pub struct ContentProcessor {
    client: PurviewClient,
}

impl ContentProcessor {
    pub fn new(client: PurviewClient) -> Self {
        Self { client }
    }

    /// Evaluate `messages` for policy violations, resolving the user id per
    /// [`resolve_user_id`] (`provided_user_id` lets a response-phase
    /// evaluation reuse the id resolved during the prompt phase, matching
    /// Python's `process_messages(..., user_id=resolved_user_id)`).
    ///
    /// Returns `(should_block, resolved_user_id)`. One `processContent` call
    /// is made per message, in order, stopping at (and including) the first
    /// one that returns a block verdict — mirrors
    /// `ScopedContentProcessor.process_messages`'s `for req in
    /// pc_requests: ...; if should_block: break`.
    ///
    /// Fails (rather than silently allowing) when `tenant_id` or
    /// `purview_app_location` aren't set on `settings`, mirroring Python's
    /// `_map_messages` raising `ValueError` in the same situation — an
    /// error a caller with `ignore_exceptions = true` treats as fail-open
    /// (see [`crate::middleware`]), same as Python.
    pub async fn evaluate(
        &self,
        messages: &[ChatMessage],
        settings: &PurviewSettings,
        provided_user_id: Option<&str>,
    ) -> Result<(bool, Option<String>)> {
        let tenant_id = settings.tenant_id.as_deref().ok_or_else(|| {
            Error::Configuration(
                "PurviewSettings::tenant_id is required (this port infers it from neither a \
                 protectionScopes precheck nor the bearer token's JWT claims)"
                    .into(),
            )
        })?;
        if !is_valid_guid(tenant_id) {
            return Err(Error::Configuration(format!(
                "PurviewSettings::tenant_id '{tenant_id}' is not a valid GUID"
            )));
        }
        let app_location = settings
            .purview_app_location
            .as_ref()
            .ok_or_else(|| {
                Error::Configuration(
                    "PurviewSettings::purview_app_location is required (this port infers it \
                     from neither a protectionScopes precheck nor the bearer token's JWT \
                     claims)"
                        .into(),
                )
            })?
            .to_policy_location();

        let Some(user_id) = resolve_user_id(messages, provided_user_id) else {
            // No resolvable user id: fail open, matching Python (an empty
            // `pc_requests` list never enters the block-checking loop).
            return Ok((false, None));
        };

        for message in messages {
            let request = build_request(
                message,
                &user_id,
                tenant_id,
                &settings.app_name,
                &app_location,
            );
            let response = self.client.process_content(&request).await?;
            if response.should_block() {
                return Ok((true, Some(user_id)));
            }
        }
        Ok((false, Some(user_id)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::types::Role;
    use std::collections::HashMap;

    fn msg_with_user_id(text: &str, user_id: &str) -> ChatMessage {
        let mut m = ChatMessage::user(text);
        let mut props = HashMap::new();
        props.insert("user_id".to_string(), serde_json::json!(user_id));
        m.additional_properties = props;
        m
    }

    fn msg_with_author(text: &str, author_name: &str) -> ChatMessage {
        ChatMessage::new(Role::user(), text).with_author(author_name)
    }

    // -- is_valid_guid ------------------------------------------------------

    #[test]
    fn is_valid_guid_accepts_well_formed_guids() {
        assert!(is_valid_guid("12345678-1234-1234-1234-123456789012"));
        assert!(is_valid_guid("a1b2c3d4-e5f6-4a5b-8c9d-0e1f2a3b4c5d"));
    }

    #[test]
    fn is_valid_guid_rejects_garbage() {
        assert!(!is_valid_guid("not-a-guid"));
        assert!(!is_valid_guid(""));
    }

    // -- resolve_user_id ------------------------------------------------------

    #[test]
    fn resolve_user_id_prefers_additional_properties_user_id() {
        let guid = "12345678-1234-1234-1234-123456789012";
        let messages = vec![msg_with_user_id("hi", guid)];
        assert_eq!(resolve_user_id(&messages, None).as_deref(), Some(guid));
    }

    #[test]
    fn resolve_user_id_falls_back_to_guid_shaped_author_name() {
        let guid = "12345678-1234-1234-1234-123456789012";
        let messages = vec![msg_with_author("hi", guid)];
        assert_eq!(resolve_user_id(&messages, None).as_deref(), Some(guid));
    }

    #[test]
    fn resolve_user_id_prefers_explicit_user_id_over_author_name_fallback() {
        let author_guid = "11111111-1111-1111-1111-111111111111";
        let user_id_guid = "22222222-2222-2222-2222-222222222222";
        let messages = vec![
            msg_with_author("first", author_guid),
            msg_with_user_id("second", user_id_guid),
        ];
        assert_eq!(
            resolve_user_id(&messages, None).as_deref(),
            Some(user_id_guid)
        );
    }

    #[test]
    fn resolve_user_id_falls_back_to_provided_when_nothing_in_messages() {
        let guid = "33333333-3333-3333-3333-333333333333";
        let messages = vec![ChatMessage::user("hi")];
        assert_eq!(
            resolve_user_id(&messages, Some(guid)).as_deref(),
            Some(guid)
        );
    }

    #[test]
    fn resolve_user_id_ignores_non_guid_provided_fallback() {
        let messages = vec![ChatMessage::user("hi")];
        assert!(resolve_user_id(&messages, Some("not-a-guid")).is_none());
    }

    #[test]
    fn resolve_user_id_none_when_nothing_resolvable() {
        let messages = vec![ChatMessage::user("hi"), ChatMessage::assistant("there")];
        assert!(resolve_user_id(&messages, None).is_none());
    }

    #[test]
    fn resolve_user_id_ignores_non_guid_user_id_property() {
        let messages = vec![msg_with_user_id("hi", "not-a-guid")];
        assert!(resolve_user_id(&messages, None).is_none());
    }

    // -- evaluate: configuration validation (async, no network) ------------

    #[tokio::test]
    async fn evaluate_fails_without_tenant_id() {
        let settings = PurviewSettings::new("App").with_purview_app_location(
            crate::settings::PurviewAppLocation::new(
                crate::settings::PurviewLocationType::Application,
                "app-1",
            ),
        );
        let processor = ContentProcessor::new(PurviewClient::new(
            crate::auth::StaticTokenProvider::new("t"),
            &settings,
        ));
        let err = processor
            .evaluate(&[ChatMessage::user("hi")], &settings, None)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("tenant_id"));
    }

    #[tokio::test]
    async fn evaluate_fails_without_app_location() {
        let settings =
            PurviewSettings::new("App").with_tenant_id("12345678-1234-1234-1234-123456789012");
        let processor = ContentProcessor::new(PurviewClient::new(
            crate::auth::StaticTokenProvider::new("t"),
            &settings,
        ));
        let err = processor
            .evaluate(&[ChatMessage::user("hi")], &settings, None)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("purview_app_location"));
    }

    #[tokio::test]
    async fn evaluate_returns_allow_without_any_network_call_when_no_user_id_resolvable() {
        // Config is valid, but no message/author/provided id is GUID-shaped
        // -- if this attempted an HTTP call, it would hang/fail trying to
        // reach graph.microsoft.com.
        let settings = PurviewSettings::new("App")
            .with_tenant_id("12345678-1234-1234-1234-123456789012")
            .with_purview_app_location(crate::settings::PurviewAppLocation::new(
                crate::settings::PurviewLocationType::Application,
                "app-1",
            ));
        let processor = ContentProcessor::new(PurviewClient::new(
            crate::auth::StaticTokenProvider::new("t"),
            &settings,
        ));
        let (should_block, user_id) = processor
            .evaluate(
                &[ChatMessage::user("hi, no identifying info here")],
                &settings,
                None,
            )
            .await
            .unwrap();
        assert!(!should_block);
        assert!(user_id.is_none());
    }
}
