//! Wire types for the Microsoft Graph `dataSecurityAndGovernance`
//! `processContent` API.
//!
//! Field names and nesting mirror Python's `agent_framework_purview._models`
//! (which in turn mirrors the Graph API's OData shape: `@odata.type`
//! discriminators, camelCase field names). Only the subset this port's
//! [`processContent`](crate::client::PurviewClient::process_content) call
//! actually sends/receives is modeled — the full Python module additionally
//! defines `protectionScopes`/`contentActivities` request/response types for
//! the two other Graph endpoints this port intentionally does not call (see
//! the crate docs' "Scope" section).

use serde::{Deserialize, Serialize};

/// High-level activity type describing what's being done with content.
/// Mirrors Python's `Activity` enum. Only `UploadText` is ever sent by this
/// port's middleware — see the crate docs for why both the prompt and
/// response checks use it (faithfully mirroring the Python reference, which
/// does the same, `DownloadText` notwithstanding).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Activity {
    Unknown,
    UploadText,
    UploadFile,
    DownloadText,
    DownloadFile,
}

/// `ActivityMetadata`: wraps an [`Activity`] for `contentToProcess`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityMetadata {
    pub activity: Activity,
}

/// `microsoft.graph.textContent`: the message text being evaluated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PurviewTextContent {
    #[serde(rename = "@odata.type")]
    pub data_type: String,
    pub data: String,
}

impl PurviewTextContent {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            data_type: "microsoft.graph.textContent".to_string(),
            data: text.into(),
        }
    }
}

/// `microsoft.graph.processConversationMetadata`: one message's content plus
/// identity metadata. `ContentToProcess::content_entries` carries a list of
/// these, though this port's [`ContentProcessor`](crate::processor::ContentProcessor)
/// only ever sends one per `processContent` call (mirrors Python's
/// `_map_messages`, which builds one whole `ProcessContentRequest` — with a
/// single-element `content_entries` — per [`Message`](agent_framework_core::types::Message),
/// not one batched request for all of them).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessConversationMetadata {
    #[serde(rename = "@odata.type")]
    pub data_type: String,
    pub identifier: String,
    pub content: PurviewTextContent,
    pub name: String,
    #[serde(rename = "isTruncated")]
    pub is_truncated: bool,
}

impl ProcessConversationMetadata {
    pub fn new(
        identifier: impl Into<String>,
        text: impl Into<String>,
        name: impl Into<String>,
    ) -> Self {
        Self {
            data_type: "microsoft.graph.processConversationMetadata".to_string(),
            identifier: identifier.into(),
            content: PurviewTextContent::new(text),
            name: name.into(),
            is_truncated: false,
        }
    }
}

/// `microsoft.graph.operatingSystemSpecifications`, nested under
/// [`DeviceMetadata`]. This port always sends `"Unknown"`/`"Unknown"`,
/// matching Python's `_map_messages` (which hardcodes the same values —
/// device introspection is out of scope for both).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatingSystemSpecifications {
    #[serde(rename = "operatingSystemPlatform")]
    pub operating_system_platform: String,
    #[serde(rename = "operatingSystemVersion")]
    pub operating_system_version: String,
}

impl Default for OperatingSystemSpecifications {
    fn default() -> Self {
        Self {
            operating_system_platform: "Unknown".to_string(),
            operating_system_version: "Unknown".to_string(),
        }
    }
}

/// `microsoft.graph.deviceMetadata`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeviceMetadata {
    #[serde(rename = "operatingSystemSpecifications")]
    pub operating_system_specifications: OperatingSystemSpecifications,
}

/// `microsoft.graph.integratedAppMetadata`: the calling application's
/// identity (name/version), independent of *where* it's deployed (see
/// [`PolicyLocation`] for that).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegratedAppMetadata {
    pub name: String,
    pub version: String,
}

/// A policy location (`@odata.type` + `value`), e.g.
/// `microsoft.graph.policyLocationApplication`. Mirrors Python's
/// `PolicyLocation`; see [`crate::settings::PurviewAppLocation::to_policy_location`]
/// for how the crate's public [`PurviewLocationType`](crate::settings::PurviewLocationType)
/// enum maps to the `@odata.type` string.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyLocation {
    #[serde(rename = "@odata.type")]
    pub data_type: String,
    pub value: String,
}

/// `microsoft.graph.protectedAppMetadata`: the app's identity plus *where*
/// it's running, for policy location matching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtectedAppMetadata {
    pub name: String,
    pub version: String,
    #[serde(rename = "applicationLocation")]
    pub application_location: PolicyLocation,
}

/// `microsoft.graph.contentToProcess`: the full bundle of content +
/// activity/device/app metadata sent to `processContent`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentToProcess {
    #[serde(rename = "contentEntries")]
    pub content_entries: Vec<ProcessConversationMetadata>,
    #[serde(rename = "activityMetadata")]
    pub activity_metadata: ActivityMetadata,
    #[serde(rename = "deviceMetadata")]
    pub device_metadata: DeviceMetadata,
    #[serde(rename = "integratedAppMetadata")]
    pub integrated_app_metadata: IntegratedAppMetadata,
    #[serde(rename = "protectedAppMetadata")]
    pub protected_app_metadata: ProtectedAppMetadata,
}

/// The `processContent` request body. Mirrors Python's
/// `ProcessContentRequest`; `scope_identifier` (sent as an `If-None-Match`
/// header, not a body field, and only meaningful after a
/// `protectionScopes/compute` precheck this port doesn't perform) and
/// `process_inline` (ditto — derived from that same precheck's execution
/// mode) are intentionally not modeled. See the crate docs' "Scope" section.
#[derive(Debug, Clone, Serialize)]
pub struct ProcessContentRequest {
    #[serde(rename = "contentToProcess")]
    pub content_to_process: ContentToProcess,
    #[serde(rename = "userId")]
    pub user_id: String,
    #[serde(rename = "tenantId")]
    pub tenant_id: String,
    #[serde(rename = "correlationId", skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<String>,
}

/// `blockAccess` vs. anything else. Mirrors Python's `DlpAction`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum DlpAction {
    BlockAccess,
    Other,
}

/// `block` vs. anything else. Mirrors Python's `RestrictionAction`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RestrictionAction {
    Block,
    Other,
}

/// One policy action returned by `processContent`. A response is a *block*
/// verdict when any entry has `action == BlockAccess` **or**
/// `restriction_action == Block` — see
/// [`ProcessContentResponse::should_block`].
#[derive(Debug, Clone, Deserialize)]
pub struct DlpActionInfo {
    #[serde(default)]
    pub action: Option<DlpAction>,
    #[serde(rename = "restrictionAction", default)]
    pub restriction_action: Option<RestrictionAction>,
}

/// Whether a protection scope's applicability has changed since it was last
/// computed/cached. This port doesn't cache (see the crate docs), so this is
/// informational only — surfaced for callers who want it, not acted on
/// internally.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ProtectionScopeState {
    NotModified,
    Modified,
    UnknownFutureValue,
}

/// One entry of `processingErrors` on a `processContent` response.
#[derive(Debug, Clone, Deserialize)]
pub struct ProcessingError {
    #[serde(default)]
    pub message: Option<String>,
}

/// The `processContent` response body. Mirrors Python's
/// `ProcessContentResponse`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ProcessContentResponse {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(rename = "protectionScopeState", default)]
    pub protection_scope_state: Option<ProtectionScopeState>,
    #[serde(rename = "policyActions", default)]
    pub policy_actions: Option<Vec<DlpActionInfo>>,
    #[serde(rename = "processingErrors", default)]
    pub processing_errors: Option<Vec<ProcessingError>>,
}

impl ProcessContentResponse {
    /// Whether this response's policy actions constitute a *block* verdict.
    ///
    /// Mirrors `ScopedContentProcessor.process_messages`'s check:
    /// `action == DlpAction.BLOCK_ACCESS or restriction_action ==
    /// RestrictionAction.BLOCK` on any entry of `policy_actions`.
    pub fn should_block(&self) -> bool {
        self.policy_actions.as_deref().is_some_and(|actions| {
            actions.iter().any(|a| {
                a.action == Some(DlpAction::BlockAccess)
                    || a.restriction_action == Some(RestrictionAction::Block)
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activity_serializes_as_lower_camel_case() {
        assert_eq!(
            serde_json::to_value(Activity::UploadText).unwrap(),
            serde_json::json!("uploadText")
        );
        assert_eq!(
            serde_json::to_value(Activity::DownloadText).unwrap(),
            serde_json::json!("downloadText")
        );
    }

    #[test]
    fn dlp_action_and_restriction_action_serialize_as_expected_wire_strings() {
        assert_eq!(
            serde_json::to_value(DlpAction::BlockAccess).unwrap(),
            serde_json::json!("blockAccess")
        );
        assert_eq!(
            serde_json::to_value(RestrictionAction::Block).unwrap(),
            serde_json::json!("block")
        );
    }

    #[test]
    fn process_content_request_serializes_with_graph_camel_case_and_odata_type() {
        let content = ContentToProcess {
            content_entries: vec![ProcessConversationMetadata::new(
                "msg-1",
                "hello",
                "Agent Framework Message msg-1",
            )],
            activity_metadata: ActivityMetadata {
                activity: Activity::UploadText,
            },
            device_metadata: DeviceMetadata::default(),
            integrated_app_metadata: IntegratedAppMetadata {
                name: "App".into(),
                version: "1.0".into(),
            },
            protected_app_metadata: ProtectedAppMetadata {
                name: "App".into(),
                version: "1.0".into(),
                application_location: PolicyLocation {
                    data_type: "microsoft.graph.policyLocationApplication".into(),
                    value: "app-id".into(),
                },
            },
        };
        let request = ProcessContentRequest {
            content_to_process: content,
            user_id: "user-123".into(),
            tenant_id: "tenant-456".into(),
            correlation_id: Some("corr-1".into()),
        };
        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(value["userId"], serde_json::json!("user-123"));
        assert_eq!(value["tenantId"], serde_json::json!("tenant-456"));
        assert_eq!(value["correlationId"], serde_json::json!("corr-1"));
        let entry = &value["contentToProcess"]["contentEntries"][0];
        assert_eq!(
            entry["@odata.type"],
            serde_json::json!("microsoft.graph.processConversationMetadata")
        );
        assert_eq!(entry["content"]["data"], serde_json::json!("hello"));
        assert_eq!(
            entry["content"]["@odata.type"],
            serde_json::json!("microsoft.graph.textContent")
        );
        assert_eq!(
            value["contentToProcess"]["activityMetadata"]["activity"],
            serde_json::json!("uploadText")
        );
        assert_eq!(
            value["contentToProcess"]["protectedAppMetadata"]["applicationLocation"]["@odata.type"],
            serde_json::json!("microsoft.graph.policyLocationApplication")
        );
        assert_eq!(
            value["contentToProcess"]["deviceMetadata"]["operatingSystemSpecifications"]
                ["operatingSystemPlatform"],
            serde_json::json!("Unknown")
        );
    }

    #[test]
    fn process_content_request_omits_correlation_id_when_none() {
        let content = ContentToProcess {
            content_entries: vec![],
            activity_metadata: ActivityMetadata {
                activity: Activity::UploadText,
            },
            device_metadata: DeviceMetadata::default(),
            integrated_app_metadata: IntegratedAppMetadata {
                name: "A".into(),
                version: "1".into(),
            },
            protected_app_metadata: ProtectedAppMetadata {
                name: "A".into(),
                version: "1".into(),
                application_location: PolicyLocation {
                    data_type: "microsoft.graph.policyLocationApplication".into(),
                    value: "v".into(),
                },
            },
        };
        let request = ProcessContentRequest {
            content_to_process: content,
            user_id: "u".into(),
            tenant_id: "t".into(),
            correlation_id: None,
        };
        let value = serde_json::to_value(&request).unwrap();
        assert!(value.get("correlationId").is_none());
    }

    // -- should_block verdict parsing --------------------------------------

    #[test]
    fn should_block_true_on_block_access_action() {
        let resp = ProcessContentResponse {
            policy_actions: Some(vec![DlpActionInfo {
                action: Some(DlpAction::BlockAccess),
                restriction_action: None,
            }]),
            ..Default::default()
        };
        assert!(resp.should_block());
    }

    #[test]
    fn should_block_true_on_block_restriction_action() {
        let resp = ProcessContentResponse {
            policy_actions: Some(vec![DlpActionInfo {
                action: None,
                restriction_action: Some(RestrictionAction::Block),
            }]),
            ..Default::default()
        };
        assert!(resp.should_block());
    }

    #[test]
    fn should_block_false_when_action_is_other() {
        let resp = ProcessContentResponse {
            policy_actions: Some(vec![DlpActionInfo {
                action: Some(DlpAction::Other),
                restriction_action: Some(RestrictionAction::Other),
            }]),
            ..Default::default()
        };
        assert!(!resp.should_block());
    }

    #[test]
    fn should_block_false_when_no_policy_actions() {
        assert!(!ProcessContentResponse::default().should_block());
        let resp = ProcessContentResponse {
            policy_actions: Some(vec![]),
            ..Default::default()
        };
        assert!(!resp.should_block());
    }

    #[test]
    fn should_block_true_when_any_of_several_actions_blocks() {
        let resp = ProcessContentResponse {
            policy_actions: Some(vec![
                DlpActionInfo {
                    action: Some(DlpAction::Other),
                    restriction_action: None,
                },
                DlpActionInfo {
                    action: Some(DlpAction::BlockAccess),
                    restriction_action: None,
                },
            ]),
            ..Default::default()
        };
        assert!(resp.should_block());
    }

    #[test]
    fn process_content_response_deserializes_from_graph_shape() {
        let value = serde_json::json!({
            "id": "resp-1",
            "protectionScopeState": "notModified",
            "policyActions": [{"action": "blockAccess", "restrictionAction": "block"}],
        });
        let resp: ProcessContentResponse = serde_json::from_value(value).unwrap();
        assert_eq!(resp.id.as_deref(), Some("resp-1"));
        assert_eq!(
            resp.protection_scope_state,
            Some(ProtectionScopeState::NotModified)
        );
        assert!(resp.should_block());
    }

    #[test]
    fn process_content_response_deserializes_empty_object() {
        let resp: ProcessContentResponse = serde_json::from_value(serde_json::json!({})).unwrap();
        assert!(resp.id.is_none());
        assert!(!resp.should_block());
    }
}
