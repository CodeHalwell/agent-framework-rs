//! [`ContentProcessor`]: maps [`Message`]s to `processContent` requests,
//! resolves the acting user id, and evaluates the resulting verdicts.
//!
//! A scoped-down port of Python's `ScopedContentProcessor` — see
//! [`crate::client`]'s module docs for exactly what's cut (the protection-
//! scopes precheck, caching, and background content-activity logging) and
//! why. What *is* ported faithfully: per-content-entry request construction
//! (the internal `build_requests`) and the GUID-based user-id resolution algorithm
//! (mirrors `ScopedContentProcessor._map_messages`'s
//! `additional_properties["user_id"]` / `author_name` scan, minus the
//! bearer-token-JWT fallback — see the crate docs).

use agent_framework_core::error::{Error, Result};
use agent_framework_core::types::{Content, Message};

use crate::client::PurviewClient;
use crate::models::{
    Activity, ActivityMetadata, ContentToProcess, DeviceMetadata, IntegratedAppMetadata,
    ProcessContentRequest, ProcessConversationMetadata, ProtectedAppMetadata, PurviewBinaryContent,
    PurviewContent, PurviewTextContent,
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
pub fn resolve_user_id(messages: &[Message], provided: Option<&str>) -> Option<String> {
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

/// Render an arbitrary content item as text a classifier can read.
///
/// The whole item is serialized rather than named fields picked out of it:
/// `additional_properties` is a data channel in its own right, and a content
/// type added after this was written must not arrive unevaluated. Mirrors
/// upstream's `_serialize_for_evaluation` (#8370).
fn serialize_for_evaluation(content: &Content) -> String {
    let mut value = match serde_json::to_value(content) {
        Ok(value) => value,
        Err(_) => return format!("{content:?}"),
    };
    // `FunctionResultContent::exception` redacts to a fixed marker when it is
    // serialized, which is right for persistence (#8235) and exactly wrong
    // here: a failing tool's diagnostic is one of the likeliest places for a
    // connection string or a quoted row to appear, which is what this check
    // exists to catch. Evaluating the marker instead would let a tool failure
    // carry data straight past the policy. The real text goes to Purview and
    // nowhere else — this string is submitted, never stored.
    if let Content::FunctionResult(result) = content {
        if let (Some(exception), Some(object)) = (&result.exception, value.as_object_mut()) {
            object.insert(
                "exception".to_string(),
                serde_json::Value::String(exception.clone()),
            );
        }
    }
    serde_json::to_string(&value).unwrap_or_else(|_| format!("{content:?}"))
}

/// The raw bytes behind a base64 `data:` URI, or `None` when it is not one.
///
/// RFC 2397 allows any number of `;parameter=value` segments between the
/// media type and `;base64`, so `data:text/plain;charset=utf-8;base64,...`
/// has to decode too — miss it and the payload goes to Purview as a base64
/// *string*, which every classifier reads as gibberish and every policy
/// passes. That is worse than sending nothing, because it looks like
/// coverage.
fn decode_data_uri(uri: &str) -> Option<Vec<u8>> {
    use base64::Engine;
    let rest = uri.strip_prefix("data:").or_else(|| {
        // Case-insensitive scheme, per the URI spec.
        uri.get(..5)
            .filter(|p| p.eq_ignore_ascii_case("data:"))
            .map(|_| &uri[5..])
    })?;
    let (metadata, payload) = rest.split_once(',')?;
    if !metadata
        .rsplit(';')
        .next()
        .is_some_and(|last| last.eq_ignore_ascii_case("base64"))
    {
        return None;
    }
    base64::engine::general_purpose::STANDARD
        .decode(payload.trim())
        .ok()
}

/// Whether an annotation list carries anything. `Some([])` is not data.
fn has_annotations(annotations: Option<&[agent_framework_core::types::Annotation]>) -> bool {
    annotations.is_some_and(|a| !a.is_empty())
}

/// Map one content item onto the Purview content entry that fits it, or
/// `None` when it carries no user data at all.
///
/// Everything a message carries is evaluated, not only its text. A tool
/// result is exactly where exfiltrated data shows up, an attachment is the
/// payload a DLP policy exists to catch, and this port had a hole of its own
/// on top of upstream's: it submitted `Message::text()`, which returns `""`
/// for a message carrying a refusal — so a whole message went unevaluated
/// because part of it was declined.
fn map_content(content: &Content) -> Option<PurviewContent> {
    match content {
        // Token counts, not user data.
        Content::Usage(_) => None,
        // Text and reasoning both carry payload *beside* their text, and the
        // text being present says nothing about whether the rest is. A
        // citation quotes its source in `snippet`, with a title and a URL; a
        // reasoning item carries `reasoning_details` in `protected_data`.
        // Streaming makes the combined shape the ordinary one rather than an
        // edge case, because `coalesce_text` folds a later fragment's payload
        // onto the accumulated text — so the test is "does anything beside
        // the text hold data", not "is the text empty".
        //
        // The plain case still sends the bare text: cheaper, and what
        // upstream does.
        Content::Text(t) => {
            if has_annotations(t.annotations.as_deref()) {
                Some(PurviewTextContent::new(serialize_for_evaluation(content)).into())
            } else {
                (!t.text.is_empty()).then(|| PurviewTextContent::new(&t.text).into())
            }
        }
        Content::TextReasoning(t) => {
            if has_annotations(t.annotations.as_deref())
                || t.protected_data.is_some()
                || t.raw_representation.is_some()
            {
                Some(PurviewTextContent::new(serialize_for_evaluation(content)).into())
            } else {
                (!t.text.is_empty()).then(|| PurviewTextContent::new(&t.text).into())
            }
        }
        Content::Data(d) => match decode_data_uri(&d.uri) {
            // Graph rejects an empty payload, and an empty one cannot violate
            // a policy.
            Some(bytes) if bytes.is_empty() => None,
            Some(bytes) => Some(PurviewBinaryContent::new(&bytes).into()),
            // Not a base64 data URI after all: evaluate its serialized form
            // rather than drop it.
            None => Some(PurviewTextContent::new(serialize_for_evaluation(content)).into()),
        },
        other => Some(PurviewTextContent::new(serialize_for_evaluation(other)).into()),
    }
}

/// Build one `processContent` request per content entry of a message.
///
/// Upstream splits the same way, keeping Graph's `contentEntries` array
/// contract with one entry in it. Device metadata is always
/// `"Unknown"`/`"Unknown"`, matching Python's hardcoded values.
fn build_requests(
    message: &Message,
    user_id: &str,
    tenant_id: &str,
    app_name: &str,
    app_location: &crate::models::PolicyLocation,
) -> Vec<ProcessContentRequest> {
    let message_id = message
        .message_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    message
        .contents
        .iter()
        .filter_map(map_content)
        .enumerate()
        .map(|(index, content)| {
            let identifier = if index == 0 {
                message_id.clone()
            } else {
                format!("{message_id}-{index}")
            };
            build_request_for(
                ProcessConversationMetadata::with_content(
                    identifier,
                    content,
                    format!("Agent Framework Message {message_id}"),
                ),
                user_id,
                tenant_id,
                app_name,
                app_location,
            )
        })
        .collect()
}

fn build_request_for(
    entry: ProcessConversationMetadata,
    user_id: &str,
    tenant_id: &str,
    app_name: &str,
    app_location: &crate::models::PolicyLocation,
) -> ProcessContentRequest {
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
        messages: &[Message],
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
            // Fail **closed**: Purview evaluates policy for a specific user,
            // so with no user there is no policy and nothing was evaluated.
            // Returning "not blocked" here reported an unevaluated message as
            // a cleared one, which is the failure mode this middleware exists
            // to prevent. Upstream changed the same behavior in #8370. A
            // deployment that would rather have availability than enforcement
            // still has `ignore_exceptions`, which now covers this case too.
            return Err(Error::Configuration(
                "no Entra user id could be resolved for the Purview request, so no policy can be \
                 evaluated. Provide one in each message's `additional_properties[\"user_id\"]` or \
                 `author_name`, pass one to the processor, or authenticate the credential as a \
                 user."
                    .into(),
            ));
        };

        for message in messages {
            for request in build_requests(
                message,
                &user_id,
                tenant_id,
                &settings.app_name,
                &app_location,
            ) {
                let response = self.client.process_content(&request).await?;
                if response.should_block() {
                    return Ok((true, Some(user_id)));
                }
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

    fn msg_with_user_id(text: &str, user_id: &str) -> Message {
        let mut m = Message::user(text);
        let mut props = HashMap::new();
        props.insert("user_id".to_string(), serde_json::json!(user_id));
        m.additional_properties = props;
        m
    }

    fn msg_with_author(text: &str, author_name: &str) -> Message {
        Message::new(Role::user(), text).with_author(author_name)
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
        let messages = vec![Message::user("hi")];
        assert_eq!(
            resolve_user_id(&messages, Some(guid)).as_deref(),
            Some(guid)
        );
    }

    #[test]
    fn resolve_user_id_ignores_non_guid_provided_fallback() {
        let messages = vec![Message::user("hi")];
        assert!(resolve_user_id(&messages, Some("not-a-guid")).is_none());
    }

    #[test]
    fn resolve_user_id_none_when_nothing_resolvable() {
        let messages = vec![Message::user("hi"), Message::assistant("there")];
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
            .evaluate(&[Message::user("hi")], &settings, None)
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
            .evaluate(&[Message::user("hi")], &settings, None)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("purview_app_location"));
    }

    #[tokio::test]
    async fn evaluate_fails_closed_without_any_network_call_when_no_user_id_resolvable() {
        // Config is valid, but no message/author/provided id is GUID-shaped.
        // This used to answer "not blocked", which reports an *unevaluated*
        // message as a cleared one. If it attempted an HTTP call instead, the
        // test would hang trying to reach graph.microsoft.com.
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
        let err = processor
            .evaluate(
                &[Message::user("hi, no identifying info here")],
                &settings,
                None,
            )
            .await
            .expect_err("no user means no policy, which is not the same as no violation");
        assert!(err.to_string().contains("user id"), "{err}");
    }

    // region: content coverage (upstream #8370)

    fn app_location() -> crate::models::PolicyLocation {
        crate::settings::PurviewAppLocation::new(
            crate::settings::PurviewLocationType::Application,
            "app-1",
        )
        .to_policy_location()
    }

    fn entries(message: &Message) -> Vec<crate::models::PurviewContent> {
        build_requests(
            message,
            "12345678-1234-1234-1234-123456789012",
            "12345678-1234-1234-1234-123456789012",
            "App",
            &app_location(),
        )
        .into_iter()
        .map(|r| r.content_to_process.content_entries[0].content.clone())
        .collect()
    }

    fn text_of(content: &crate::models::PurviewContent) -> String {
        match content {
            crate::models::PurviewContent::Text(t) => t.data.clone(),
            crate::models::PurviewContent::Binary(_) => panic!("expected text content"),
        }
    }

    #[test]
    fn a_tool_result_is_evaluated_rather_than_skipped() {
        // A tool result is exactly where exfiltrated data shows up, and it
        // is not part of `Message::text()`.
        let message = Message::with_contents(
            Role::assistant(),
            vec![
                Content::text("here you go"),
                Content::FunctionResult(agent_framework_core::types::FunctionResultContent::new(
                    "c1",
                    Some(serde_json::json!({ "ssn": "123-45-6789" })),
                )),
            ],
        );
        let mapped = entries(&message);
        assert_eq!(mapped.len(), 2);
        assert_eq!(text_of(&mapped[0]), "here you go");
        assert!(text_of(&mapped[1]).contains("123-45-6789"));
    }

    #[test]
    fn a_message_carrying_a_refusal_still_has_its_text_evaluated() {
        // `Message::text()` returns "" whenever a refusal is present, so
        // submitting the message text alone sent *nothing* for a partly
        // declined turn — a hole this port had on top of upstream's.
        let message = Message::with_contents(
            Role::assistant(),
            vec![
                Content::text("the part I did answer"),
                Content::Text(agent_framework_core::types::TextContent::refusal(
                    "I can't do the rest",
                )),
            ],
        );
        assert_eq!(message.text(), "", "the premise of this test");
        let mapped = entries(&message);
        assert_eq!(mapped.len(), 2);
        assert_eq!(text_of(&mapped[0]), "the part I did answer");
    }

    #[test]
    fn a_tool_failures_diagnostic_reaches_purview_despite_being_redacted_on_disk() {
        // `FunctionResultContent::exception` serializes to a fixed marker so
        // it is never persisted (#8235). That redaction must not reach the
        // DLP check: a failing tool's diagnostic is one of the likeliest
        // places for a connection string or a quoted row to surface, which is
        // precisely what this middleware exists to catch.
        let mut result = agent_framework_core::types::FunctionResultContent::new("c1", None);
        result.exception =
            Some("connect failed: Server=db1;Password=hunter2 while reading row 17".into());
        let message =
            Message::with_contents(Role::tool(), vec![Content::FunctionResult(result.clone())]);
        let submitted = text_of(&entries(&message)[0]);
        assert!(submitted.contains("hunter2"), "{submitted}");
        assert!(
            !submitted.contains(agent_framework_core::types::FUNCTION_INVOCATION_ERROR_MARKER),
            "the marker must not stand in for the text here: {submitted}"
        );

        // And the redaction still holds everywhere else.
        let persisted = serde_json::to_string(&result).unwrap();
        assert!(!persisted.contains("hunter2"), "{persisted}");
    }

    #[test]
    fn a_citations_snippet_is_evaluated_along_with_the_text() {
        // A citation quotes its source. If the quoted span is the sensitive
        // part, sending only the surrounding prose evaluates everything
        // except the thing that matters.
        let text = agent_framework_core::types::TextContent {
            text: "see the reference".to_string(),
            annotations: Some(vec![agent_framework_core::types::Annotation {
                snippet: Some("patient MRN 55512345".to_string()),
                ..Default::default()
            }]),
            ..Default::default()
        };
        let message = Message::with_contents(Role::assistant(), vec![Content::Text(text)]);
        let submitted = text_of(&entries(&message)[0]);
        assert!(submitted.contains("55512345"), "{submitted}");
        assert!(submitted.contains("see the reference"), "{submitted}");
    }

    #[test]
    fn plain_text_is_still_submitted_as_itself() {
        // The common case stays cheap and identical to upstream: no JSON
        // wrapper when there is nothing but text.
        let message = Message::with_contents(Role::user(), vec![Content::text("just words")]);
        assert_eq!(text_of(&entries(&message)[0]), "just words");
    }

    #[test]
    fn reasoning_whose_payload_is_not_its_text_is_still_evaluated() {
        // Chat Completions' `reasoning_details` rides in `protected_data`
        // with an empty `text`, so an `is_empty()` check on the text alone
        // dropped a content type this same change introduced.
        let reasoning = agent_framework_core::types::TextReasoningContent {
            protected_data: Some(
                r#"[{"type":"reasoning.text","text":"the account number is 12345"}]"#.into(),
            ),
            ..Default::default()
        };
        let message =
            Message::with_contents(Role::assistant(), vec![Content::TextReasoning(reasoning)]);
        let mapped = entries(&message);
        assert_eq!(mapped.len(), 1);
        assert!(text_of(&mapped[0]).contains("12345"));
    }

    #[test]
    fn reasoning_with_both_text_and_payload_submits_both() {
        // The ordinary post-streaming shape: `coalesce_text` folds a later
        // fragment's `protected_data` onto the accumulated text, so a summary
        // being present is no reason to stop looking at the payload beside
        // it.
        let reasoning = agent_framework_core::types::TextReasoningContent {
            text: "weighing the options".to_string(),
            protected_data: Some(
                r#"[{"type":"reasoning.text","text":"the account number is 12345"}]"#.into(),
            ),
            ..Default::default()
        };
        let message =
            Message::with_contents(Role::assistant(), vec![Content::TextReasoning(reasoning)]);
        let submitted = text_of(&entries(&message)[0]);
        assert!(submitted.contains("12345"), "{submitted}");
        assert!(submitted.contains("weighing the options"), "{submitted}");
    }

    #[test]
    fn reasoning_with_only_text_is_submitted_as_itself() {
        let reasoning = agent_framework_core::types::TextReasoningContent {
            text: "just thinking".to_string(),
            ..Default::default()
        };
        let message =
            Message::with_contents(Role::assistant(), vec![Content::TextReasoning(reasoning)]);
        assert_eq!(text_of(&entries(&message)[0]), "just thinking");
    }

    #[test]
    fn an_empty_annotation_list_is_not_data() {
        // `Some([])` must not push the plain case onto the serializing path.
        let text = agent_framework_core::types::TextContent {
            text: "plain".to_string(),
            annotations: Some(Vec::new()),
            ..Default::default()
        };
        let message = Message::with_contents(Role::user(), vec![Content::Text(text)]);
        assert_eq!(text_of(&entries(&message)[0]), "plain");
    }

    #[test]
    fn reasoning_with_neither_text_nor_payload_has_nothing_to_evaluate() {
        let message = Message::with_contents(
            Role::assistant(),
            vec![Content::TextReasoning(Default::default())],
        );
        assert!(entries(&message).is_empty());
    }

    #[test]
    fn an_attachment_is_sent_as_bytes_not_as_its_base64_text() {
        // A classifier reads bytes; handed the base64 string it reads
        // gibberish and passes every policy, which looks like coverage.
        let message = Message::with_contents(
            Role::user(),
            // "secret" in base64, with a parameterised media type — RFC
            // 2397 allows any number of `;parameter=value` segments, and
            // missing them is what sends the payload as text.
            vec![Content::Data(agent_framework_core::types::DataContent {
                uri: "data:text/plain;charset=utf-8;base64,c2VjcmV0".to_string(),
                media_type: Some("text/plain".to_string()),
            })],
        );
        let mapped = entries(&message);
        assert_eq!(mapped.len(), 1);
        match &mapped[0] {
            crate::models::PurviewContent::Binary(b) => assert_eq!(b.data, "c2VjcmV0"),
            other => panic!("expected binary content, got {other:?}"),
        }
    }

    #[test]
    fn a_non_base64_uri_is_evaluated_as_text_rather_than_dropped() {
        let message = Message::with_contents(
            Role::user(),
            vec![Content::Data(agent_framework_core::types::DataContent {
                uri: "https://example.com/secret-report.pdf".to_string(),
                media_type: None,
            })],
        );
        assert!(text_of(&entries(&message)[0]).contains("secret-report.pdf"));
    }

    #[test]
    fn usage_is_the_only_content_with_nothing_to_evaluate() {
        let message = Message::with_contents(
            Role::assistant(),
            vec![
                Content::Usage(agent_framework_core::types::UsageContent {
                    details: agent_framework_core::types::UsageDetails::new(),
                }),
                // Empty text has nothing to classify and Graph rejects it.
                Content::text(""),
            ],
        );
        assert!(entries(&message).is_empty());
    }

    #[test]
    fn each_content_entry_gets_its_own_request_with_a_distinct_identifier() {
        let mut message = Message::with_contents(
            Role::user(),
            vec![Content::text("one"), Content::text("two")],
        );
        message.message_id = Some("m1".to_string());
        let requests = build_requests(
            &message,
            "12345678-1234-1234-1234-123456789012",
            "12345678-1234-1234-1234-123456789012",
            "App",
            &app_location(),
        );
        let ids: Vec<_> = requests
            .iter()
            .map(|r| r.content_to_process.content_entries[0].identifier.clone())
            .collect();
        assert_eq!(ids, vec!["m1".to_string(), "m1-1".to_string()]);
    }
}
