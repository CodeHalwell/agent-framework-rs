//! The typed Voice relay (feature `voice`): the codec, ported from
//! upstream's `tests/voice/test_voice_codec.py` and
//! `test_voice_protocol_findings.py`, and the host over a real WebSocket,
//! from `test_voice_host.py` and the sample's handshake.
#![cfg(feature = "voice")]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_framework_core::error::{Error, Result};
use agent_framework_foundry_hosting::voice::*;
use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio_tungstenite::tungstenite::{self, client::IntoClientRequest};

// ---------------------------------------------------------------------------
// Helpers.

const TS: &str = "2026-08-12T00:00:00Z";

fn frame(message_type: &str, fields: Value) -> String {
    let mut payload = json!({"type": message_type, "id": "m_1", "ts": TS});
    if let Value::Object(extra) = fields {
        payload.as_object_mut().unwrap().extend(extra);
    }
    payload.to_string()
}

fn session_start(version: &str) -> String {
    frame(
        "session.start",
        json!({
            "protocol_version": version,
            "reconnect": false,
            "response_timeouts": {"first_output_ms": 1, "idle_ms": 2, "max_duration_ms": 3},
        }),
    )
}

fn decode(text: &str) -> std::result::Result<Option<InboundVoiceMessage>, VoiceProtocolError> {
    decode_inbound_message(text)
}

fn decode_err(text: &str) -> VoiceProtocolError {
    decode(text).expect_err("a protocol error")
}

fn encode(message: impl Into<OutboundVoiceMessage>) -> std::result::Result<Value, VoiceError> {
    encode_outbound_message(&message.into()).map(|f| serde_json::from_str(&f).unwrap())
}

fn encode_err(message: impl Into<OutboundVoiceMessage>) -> String {
    encode(message).expect_err("an encoding error").to_string()
}

fn object(value: Value) -> JsonObject {
    match value {
        Value::Object(map) => map,
        other => panic!("not an object: {other}"),
    }
}

// ---------------------------------------------------------------------------
// Decoding (test_voice_codec.py).

#[test]
fn decode_preserves_mixed_content_and_data() {
    let event = decode(&frame(
        "user.message",
        json!({"item_id": "in_data", "content": [
            {"type": "input_text", "text": "hello"},
            {"type": "input_data", "data": {"nested": [1, true, null, {"key": "value"}]}},
        ]}),
    ))
    .unwrap();
    let Some(InboundVoiceMessage::UserMessage(message)) = event else {
        panic!("{event:?}")
    };
    assert_eq!(message.item_id, "in_data");
    assert_eq!(
        message.content,
        vec![
            InputContentPart::InputText {
                text: "hello".into()
            },
            InputContentPart::InputData {
                data: object(json!({"nested": [1, true, null, {"key": "value"}]}))
            },
        ]
    );
}

#[test]
fn decode_rejects_non_object_data() {
    for payload in [json!(null), json!([]), json!("text"), json!(1)] {
        let error = decode_err(&frame(
            "user.message",
            json!({"item_id": "in_data", "content": [{"type": "input_data", "data": payload}]}),
        ));
        assert_eq!(error.message(), "data must be an object");
        assert_eq!(error.close_code(), 1002);
    }
}

#[test]
fn empty_data_object_is_valid_both_ways() {
    let event = decode(&frame(
        "user.message",
        json!({"item_id": "in_data", "content": [{"type": "input_data", "data": {}}]}),
    ))
    .unwrap();
    assert!(matches!(event, Some(InboundVoiceMessage::UserMessage(m))
        if m.content == vec![InputContentPart::InputData { data: JsonObject::new() }]));
    let encoded = encode(ResponseOutputDataDone::new(
        "r_data",
        "it_data",
        JsonObject::new(),
    ))
    .unwrap();
    assert_eq!(encoded["data"], json!({}));
}

#[test]
fn decode_all_selected_inbound_messages() {
    let frames = [
        (session_start("1.0"), "session.start"),
        (
            frame(
                "user.message",
                json!({"item_id": "in_1", "content": [{"type": "input_text", "text": "hello"}]}),
            ),
            "user.message",
        ),
        (
            frame("user.no_input", json!({"item_id": "in_2", "count": 1})),
            "user.no_input",
        ),
        (
            frame("user.speech_started", json!({})),
            "user.speech_started",
        ),
        (
            frame(
                "user.barge_in",
                json!({"response_id": "r_1", "heard_text": "heard"}),
            ),
            "user.barge_in",
        ),
        (
            frame("response.accepted", json!({"response_id": "r_2"})),
            "response.accepted",
        ),
        (
            frame(
                "response.dropped",
                json!({"response_id": "r_3", "reason": "queue_full"}),
            ),
            "response.dropped",
        ),
        (
            frame(
                "response.cancelled",
                json!({"response_id": "r_4", "heard_text": "heard"}),
            ),
            "response.cancelled",
        ),
        (
            frame(
                "response.timeout",
                json!({"item_ids": ["in_3"], "stage": "first_output"}),
            ),
            "response.timeout",
        ),
        (
            frame("session.end", json!({"reason": "caller_hangup"})),
            "session.end",
        ),
    ];
    for (text, expected) in frames {
        let event = decode(&text).unwrap().expect("a selected message");
        assert_eq!(event.message_type(), expected);
        assert_eq!(event.id(), "m_1");
        // The typed model re-serializes to the same wire object.
        let wire: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(serde_json::to_value(&event).unwrap(), wire, "{expected}");
    }
}

#[test]
fn user_barge_in_decodes_with_playback_snapshot() {
    let event = decode(&frame(
        "user.barge_in",
        json!({"response_id": "r_1", "item_id": "it_1", "heard_text": "heard"}),
    ))
    .unwrap();
    assert_eq!(
        event,
        Some(InboundVoiceMessage::UserBargeIn(UserBargeIn {
            id: "m_1".into(),
            ts: TS.into(),
            response_id: "r_1".into(),
            heard_text: "heard".into(),
            item_id: Some("it_1".into()),
        }))
    );
}

#[test]
fn legacy_barge_in_and_unknown_messages_are_ignored() {
    assert_eq!(
        decode(&frame(
            "barge_in",
            json!({"response_id": "r_1", "heard_text": "heard"})
        ))
        .unwrap(),
        None
    );
    assert_eq!(
        decode(&frame("future.message", json!({"value": 1}))).unwrap(),
        None
    );
}

#[test]
fn session_start_keeps_caller_context() {
    let event = decode(&frame(
        "session.start",
        json!({
            "protocol_version": "1.0",
            "reconnect": false,
            "response_timeouts": {"first_output_ms": 1, "idle_ms": 2, "max_duration_ms": 3},
            "caller": {"custom_parameters": {"campaign": ["renewal"]}},
        }),
    ))
    .unwrap();
    let Some(InboundVoiceMessage::SessionStart(start)) = event else {
        panic!()
    };
    assert_eq!(
        start.caller,
        Some(object(
            json!({"custom_parameters": {"campaign": ["renewal"]}})
        ))
    );
    assert_eq!(
        start.response_timeouts,
        ResponseTimeouts {
            first_output_ms: 1,
            idle_ms: 2,
            max_duration_ms: 3
        }
    );
}

#[test]
fn session_start_rejects_credential_bearing_caller_context() {
    let callers = [
        json!({"authorization": "Bearer customer-secret"}),
        json!({"custom_parameters": {"api_key": "customer-secret"}}),
        json!({"future_context": [{"client-secret": "customer-secret"}]}),
        json!({"future_context": {"proxy_authorization": "Bearer customer-secret"}}),
        json!({"future_context": {"azureApiKey": "customer-secret"}}),
        json!({"future_context": {"bearer_token": "customer-secret"}}),
        json!({"future_context": {"auth_token": "customer-secret"}}),
        json!({"future_context": {"apiToken": "customer-secret"}}),
        json!({"future_context": {"secretValue": "customer-secret"}}),
        json!({"future_context": {"authorizationHeader": "Bearer customer-secret"}}),
        json!({"future_context": {"sas_url": "https://example.test/?sig=customer-secret"}}),
        json!({"future_context": {"APIKey": "customer-secret"}}),
        json!({"future_context": {"APIKEY": "customer-secret"}}),
        json!({"future_context": {"SASUrl": "https://example.test/?sig=customer-secret"}}),
        json!({"future_context": {"SASURL": "https://example.test/?sig=customer-secret"}}),
        json!({"future_context": {"ClientSecret": "customer-secret"}}),
        json!({"future_context": {"CLIENTSECRET": "customer-secret"}}),
        json!({"future_context": {"AUTHORIZATIONHEADER": "Bearer customer-secret"}}),
        json!({"future_context": {"AZUREAPIKEY": "customer-secret"}}),
        json!({"future_context": {"PROXYAUTHORIZATION": "Bearer customer-secret"}}),
        json!({"future_context": {"XAPITOKEN": "customer-secret"}}),
        json!({"future_context": {"SERVICECLIENTSECRET": "customer-secret"}}),
        json!({"future_context": {"PROXYSECRET": "customer-secret"}}),
        json!({"future_context": {"AZURESAS": "customer-secret"}}),
        json!({"future_context": {"AZUREPWD": "customer-secret"}}),
        json!({"future_context": {"password_count": 1}}),
        json!({"future_context": {"password-count": 1}}),
        json!({"future_context": {"passwordCount": 1}}),
        json!({"future_context": {"PasswordCount": 1}}),
        json!({"future_context": {"PASSWORDCOUNT": 1}}),
    ];
    for caller in callers {
        let error = decode_err(&frame(
            "session.start",
            json!({
                "protocol_version": "1.0",
                "reconnect": false,
                "response_timeouts": {"first_output_ms": 1, "idle_ms": 2, "max_duration_ms": 3},
                "caller": caller.clone(),
            }),
        ));
        assert_eq!(
            error.message(),
            "session.start caller must not contain credentials",
            "{caller}"
        );
    }
}

#[test]
fn session_start_preserves_safe_open_caller_context() {
    let caller = json!({"future_context": {
        "token_count": 7,
        "secretary_name": "Ada",
        "SECRETARYNAME": "Grace",
        "apiVersion": "2026-08-13",
        "APIStatus": "available",
        "KANSAS": "Wichita",
        "sessionId": "session-1",
        "sasquatch": true,
    }});
    let event = decode(&frame(
        "session.start",
        json!({
            "protocol_version": "1.0",
            "reconnect": false,
            "response_timeouts": {"first_output_ms": 1, "idle_ms": 2, "max_duration_ms": 3},
            "caller": caller.clone(),
        }),
    ))
    .unwrap();
    let Some(InboundVoiceMessage::SessionStart(start)) = event else {
        panic!()
    };
    assert_eq!(start.caller, Some(object(caller)));
}

#[test]
fn session_start_rejects_invalid_unicode_in_caller_key() {
    let text = r#"{"type":"session.start","id":"m_1","ts":"2026-08-12T00:00:00Z","protocol_version":"1.0","reconnect":false,"response_timeouts":{"first_output_ms":1,"idle_ms":2,"max_duration_ms":3},"caller":{"\ud800":"value"}}"#;
    assert_eq!(
        decode_err(text).message(),
        "Voice frame contains invalid Unicode"
    );
}

#[test]
fn session_start_greeting_rules() {
    for reconnect in [false, true] {
        let error = decode_err(&frame(
            "session.start",
            json!({
                "protocol_version": "1.0",
                "reconnect": reconnect,
                "greeting": null,
                "response_timeouts": {"first_output_ms": 1, "idle_ms": 2, "max_duration_ms": 3},
            }),
        ));
        assert!(error.message().contains("greeting"), "{error}");
    }
    let error = decode_err(&frame(
        "session.start",
        json!({
            "protocol_version": "1.0",
            "reconnect": true,
            "greeting": "welcome back",
            "response_timeouts": {"first_output_ms": 1, "idle_ms": 2, "max_duration_ms": 3},
        }),
    ));
    assert_eq!(
        error.message(),
        "session.start greeting must be absent on reconnect"
    );
}

#[test]
fn session_start_field_errors_use_upstream_messages() {
    let cases = [
        (
            json!({"reconnect": false, "response_timeouts": {"first_output_ms": 1, "idle_ms": 2, "max_duration_ms": 3}}),
            "protocol_version is required",
        ),
        (
            json!({"protocol_version": "1.0", "reconnect": false}),
            "response_timeouts must be an object",
        ),
        (
            json!({"protocol_version": "1.0", "reconnect": "no", "response_timeouts": {}}),
            "session.start reconnect must be a boolean",
        ),
        (
            json!({"protocol_version": "", "reconnect": false, "response_timeouts": {"first_output_ms": 1, "idle_ms": 2, "max_duration_ms": 3}}),
            "protocol_version must be non-empty",
        ),
        (
            json!({"protocol_version": "1.0", "reconnect": false, "response_timeouts": {"first_output_ms": 0, "idle_ms": 2, "max_duration_ms": 3}}),
            "first_output_ms must be positive",
        ),
        (
            json!({"protocol_version": "1.0", "reconnect": false, "response_timeouts": {"first_output_ms": 1.5, "idle_ms": 2, "max_duration_ms": 3}}),
            "first_output_ms must be an integer",
        ),
        (
            json!({"protocol_version": "1.0", "reconnect": false, "response_timeouts": {"first_output_ms": true, "idle_ms": 2, "max_duration_ms": 3}}),
            "first_output_ms must be an integer",
        ),
        (
            json!({"protocol_version": "1.0", "reconnect": false, "response_timeouts": {"first_output_ms": 1, "max_duration_ms": 3}}),
            "idle_ms is required",
        ),
        (
            json!({"protocol_version": "1.0", "reconnect": false, "no_input_timeout_ms": -5, "response_timeouts": {"first_output_ms": 1, "idle_ms": 2, "max_duration_ms": 3}}),
            "no_input_timeout_ms must be positive",
        ),
    ];
    for (fields, expected) in cases {
        assert_eq!(
            decode_err(&frame("session.start", fields)).message(),
            expected
        );
    }
}

#[test]
fn known_excluded_messages_and_images_fail_loud() {
    for message_type in [
        "conversation.item.create",
        "conversation.item.delete",
        "dtmf",
        "dtmf.collect.cancelled",
        "dtmf.collect.rejected",
        "handoff.failed",
    ] {
        let error = decode_err(&frame(message_type, json!({})));
        assert!(error.is_unsupported());
        assert_eq!(error.close_code(), 1003);
        assert_eq!(
            error.message(),
            "Voice message type is not supported by this profile"
        );
    }
    let error = decode_err(&frame(
        "user.message",
        json!({"item_id": "in_1", "content": [
            {"type": "input_image", "image_ref": "https://example/image", "mime_type": "image/png"}
        ]}),
    ));
    assert!(error.is_unsupported());
    assert_eq!(
        error.message(),
        "Image content is not supported by this profile"
    );
}

#[test]
fn invalid_inbound_frames_fail_context_free() {
    let cases = [
        ("[]".to_string(), "Voice frame must be a JSON object"),
        (
            r#"{"type":"session.end","type":"session.end","id":"m_1","ts":"2026-08-12T00:00:00Z"}"#
                .to_string(),
            "Voice frame is not valid JSON",
        ),
        ("not-json".to_string(), "Voice frame is not valid JSON"),
        (
            frame("response.done", json!({"response_id": "r_1"})),
            "Agent-to-Bridge message received in the inbound direction",
        ),
        (
            frame(
                "response.timeout",
                json!({"response_id": "r_1", "item_ids": ["in_1"], "stage": "idle"}),
            ),
            "response.timeout requires exactly one target",
        ),
        (
            frame(
                "response.timeout",
                json!({"item_ids": [1], "stage": "idle"}),
            ),
            "item_ids must be a string",
        ),
        (
            frame(
                "response.timeout",
                json!({"item_ids": ["in_1", "in_1"], "stage": "idle"}),
            ),
            "response.timeout item_ids must be unique",
        ),
        (
            frame("response.timeout", json!({"item_ids": [], "stage": "idle"})),
            "response.timeout item_ids must be a non-empty array",
        ),
        (
            frame(
                "session.end",
                json!({"reason": "caller_hangup", "ts": "not-a-timestamp"}),
            ),
            "ts must match the RFC 3339 profile",
        ),
        (
            json!({"type": "session.end", "id": "m_1", "reason": "x"}).to_string(),
            "ts must be a string",
        ),
        (
            json!({"id": "m_1", "ts": TS}).to_string(),
            "type is required",
        ),
        (
            frame("session.end", json!({"id": "x".repeat(257), "reason": "x"})),
            "id exceeds the maximum identifier size",
        ),
        (
            frame("user.no_input", json!({"item_id": "x_1", "count": 1})),
            "item_id must start with in_",
        ),
        (
            frame("user.no_input", json!({"item_id": "in_", "count": 1})),
            "item_id must start with in_",
        ),
        (
            frame("session.end", json!({"reason": ""})),
            "reason must be non-empty",
        ),
        (
            frame("user.message", json!({"item_id": "in_1", "content": []})),
            "user.message content must be a non-empty array",
        ),
        (
            frame(
                "user.message",
                json!({"item_id": "in_1", "content": ["text"]}),
            ),
            "user.message content parts must be objects",
        ),
        (
            frame(
                "user.message",
                json!({"item_id": "in_1", "content": [{"text": "x"}]}),
            ),
            "user.message content part type must be a string",
        ),
        (
            format!(
                r#"{{"type":"session.end","id":"m_1","ts":"{TS}","reason":"x","n":{}}}"#,
                "9".repeat(129)
            ),
            "Voice frame is not valid JSON",
        ),
        (
            format!(r#"{{"type":"session.end","id":"m_1","ts":"{TS}","reason":"x","n":1e400}}"#),
            "Voice frame contains a non-finite number",
        ),
    ];
    for (text, expected) in cases {
        let error = decode_err(&text);
        assert_eq!(error.message(), expected, "{text}");
        assert_eq!(error.close_code(), 1002, "{text}");
    }
}

#[test]
fn json_limits_are_enforced() {
    let mut deep = json!(1);
    for _ in 0..32 {
        deep = json!([deep]);
    }
    let error = decode_err(&frame("session.end", json!({"reason": "x", "deep": deep})));
    assert_eq!(
        error.message(),
        "Voice frame exceeds the maximum JSON depth"
    );
    let wide: Vec<u8> = vec![0; 8_200];
    let error = decode_err(&frame("session.end", json!({"reason": "x", "wide": wide})));
    assert_eq!(
        error.message(),
        "Voice frame exceeds the maximum JSON node count"
    );
    let error = decode_err(&"x".repeat(MAX_FRAME_BYTES + 1));
    assert_eq!(
        (error.message(), error.close_code()),
        ("Voice frame exceeds the maximum size", 1009)
    );
}

#[test]
fn user_message_with_only_unknown_parts_is_ignored() {
    assert_eq!(
        decode(&frame(
            "user.message",
            json!({"item_id": "in_1", "content": [{"type": "future_part"}]})
        ))
        .unwrap(),
        None
    );
}

#[test]
fn timestamp_rejects_non_ascii_digits() {
    for ts in [
        "٢٠٢٦-08-12T00:00:00Z",
        "2026-08-12T00:00:00.١Z",
        "2026-08-12T00:00:00+٠١:00",
    ] {
        let error = decode_err(&frame(
            "session.end",
            json!({"reason": "completed", "ts": ts}),
        ));
        assert!(error.message().contains("RFC 3339"), "{ts}");
    }
    let error = decode_err(&frame(
        "session.end",
        json!({"reason": "x", "ts": "2026-02-30T00:00:00Z"}),
    ));
    assert_eq!(
        error.message(),
        "ts must contain a valid calendar timestamp"
    );
}

// ---------------------------------------------------------------------------
// Encoding.

#[test]
fn encode_all_selected_outbound_messages() {
    let messages: Vec<OutboundVoiceMessage> = vec![
        SessionReady::new().into(),
        SessionRejected::new("startup_failed", false).into(),
        ResponseCreated::reply("r_1", ["in_1"]).into(),
        ResponseNone::new(["in_1"]).into(),
        ResponseOutputTextDelta::new("r_1", "it_1", "hel").into(),
        ResponseOutputTextDone::new("r_1", "it_1", "hello").into(),
        ResponseOutputDataDone::new("r_1", "it_1", JsonObject::new()).into(),
        ResponseDone::new("r_1").into(),
        ResponseCancel::new("r_1").into(),
        EndCall::new("completed").into(),
        AgentError::new("backend_error", "failed").into(),
    ];
    let mut types = std::collections::BTreeSet::new();
    for message in messages {
        let payload = encode(message.clone()).unwrap();
        assert_eq!(payload["type"], message.message_type());
        assert!(payload["id"].as_str().unwrap().starts_with("m_"));
        assert!(payload["ts"].as_str().unwrap().ends_with('Z'));
        types.insert(message.message_type());
    }
    assert_eq!(types.len(), 11);
}

#[test]
fn outbound_wire_encodings_are_exact() {
    fn fixed<T: Into<OutboundVoiceMessage>>(message: T) -> String {
        let mut message = message.into();
        let (id, ts) = match &mut message {
            OutboundVoiceMessage::SessionReady(m) => (&mut m.id, &mut m.ts),
            OutboundVoiceMessage::SessionRejected(m) => (&mut m.id, &mut m.ts),
            OutboundVoiceMessage::ResponseCreated(m) => (&mut m.id, &mut m.ts),
            OutboundVoiceMessage::ResponseNone(m) => (&mut m.id, &mut m.ts),
            OutboundVoiceMessage::ResponseOutputDataDone(m) => (&mut m.id, &mut m.ts),
            OutboundVoiceMessage::ResponseOutputTextDelta(m) => (&mut m.id, &mut m.ts),
            OutboundVoiceMessage::ResponseOutputTextDone(m) => (&mut m.id, &mut m.ts),
            OutboundVoiceMessage::ResponseDone(m) => (&mut m.id, &mut m.ts),
            OutboundVoiceMessage::ResponseCancel(m) => (&mut m.id, &mut m.ts),
            OutboundVoiceMessage::EndCall(m) => (&mut m.id, &mut m.ts),
            OutboundVoiceMessage::Error(m) => (&mut m.id, &mut m.ts),
        };
        *id = "m_1".into();
        *ts = TS.into();
        encode_outbound_message(&message).unwrap()
    }
    let head = r#""id":"m_1","ts":"2026-08-12T00:00:00Z""#;
    assert_eq!(
        fixed(SessionReady::new()),
        format!(r#"{{"type":"session.ready",{head}}}"#)
    );
    assert_eq!(
        fixed(SessionRejected::new("protocol_mismatch", false).with_message("bad")),
        format!(
            r#"{{"type":"session.rejected",{head},"code":"protocol_mismatch","retriable":false,"message":"bad"}}"#
        )
    );
    assert_eq!(
        fixed(ResponseCreated::reply("r_1", ["in_1", "in_2"])),
        format!(
            r#"{{"type":"response.created",{head},"response_id":"r_1","in_reply_to":["in_1","in_2"]}}"#
        )
    );
    assert_eq!(
        fixed(
            ResponseCreated::proactive("r_1")
                .with_admission_timeout_ms(500)
                .with_supersede_key("k")
        ),
        format!(
            r#"{{"type":"response.created",{head},"response_id":"r_1","admission_timeout_ms":500,"supersede_key":"k"}}"#
        )
    );
    assert_eq!(
        fixed(ResponseNone::new(["in_1"]).with_reason("acknowledged")),
        format!(
            r#"{{"type":"response.none",{head},"in_reply_to":["in_1"],"reason":"acknowledged"}}"#
        )
    );
    assert_eq!(
        fixed(ResponseOutputTextDelta::new("r_1", "it_1", "hé")),
        format!(
            r#"{{"type":"response.output_text.delta",{head},"response_id":"r_1","item_id":"it_1","delta":"hé"}}"#
        )
    );
    assert_eq!(
        fixed(ResponseOutputTextDone::new("r_1", "it_1", "")),
        format!(
            r#"{{"type":"response.output_text.done",{head},"response_id":"r_1","item_id":"it_1","text":""}}"#
        )
    );
    assert_eq!(
        fixed(ResponseOutputDataDone::new(
            "r_1",
            "it_1",
            object(json!({"k": [1, null]}))
        )),
        format!(
            r#"{{"type":"response.output_data.done",{head},"response_id":"r_1","item_id":"it_1","data":{{"k":[1,null]}}}}"#
        )
    );
    assert_eq!(
        fixed(ResponseDone::new("r_1")),
        format!(r#"{{"type":"response.done",{head},"response_id":"r_1"}}"#)
    );
    assert_eq!(
        fixed(ResponseCancel::new("r_1").with_reason("superseded")),
        format!(r#"{{"type":"response.cancel",{head},"response_id":"r_1","reason":"superseded"}}"#)
    );
    assert_eq!(
        fixed(EndCall::new("completed")),
        format!(r#"{{"type":"end_call",{head},"reason":"completed","mode":"drain"}}"#)
    );
    assert_eq!(
        fixed(EndCall::new("transfer").with_mode(EndCallMode::Immediate)),
        format!(r#"{{"type":"end_call",{head},"reason":"transfer","mode":"immediate"}}"#)
    );
    assert_eq!(
        fixed(
            AgentError::new("backend_error", "failed")
                .with_response_id("r_1")
                .with_item_id("it_1")
        ),
        format!(
            r#"{{"type":"error",{head},"code":"backend_error","message":"failed","response_id":"r_1","item_id":"it_1"}}"#
        )
    );
}

#[test]
fn outbound_validation_uses_upstream_messages() {
    assert_eq!(
        encode_err(ResponseCreated::reply("r_1", ["in_1"]).with_admission_timeout_ms(5)),
        "reply response.created cannot contain proactive admission controls"
    );
    assert_eq!(
        encode_err(ResponseCreated::proactive("r_too_late").with_admission_timeout_ms(60_001)),
        "admission_timeout_ms must be at most 60000"
    );
    assert_eq!(
        encode(ResponseCreated::proactive("r_boundary").with_admission_timeout_ms(60_000)).unwrap()
            ["admission_timeout_ms"],
        60_000
    );
    assert_eq!(
        encode_err(ResponseCreated::proactive("r_1").with_admission_timeout_ms(0)),
        "admission_timeout_ms must be positive"
    );
    assert_eq!(
        encode_err(ResponseCreated::proactive("r_1").with_supersede_key("")),
        "supersede_key must be non-empty"
    );
    assert_eq!(
        encode_err(ResponseCreated::proactive("x_1")),
        "response_id must start with r_"
    );
    assert_eq!(
        encode_err(ResponseNone::new(Vec::<String>::new())),
        "in_reply_to must be a non-empty tuple"
    );
    assert_eq!(
        encode_err(ResponseNone::new(["in_1", "in_1"])),
        "in_reply_to must contain unique identifiers"
    );
    assert_eq!(
        encode_err(ResponseNone::new(["it_1"])),
        "in_reply_to must start with in_"
    );
    assert_eq!(
        encode_err(ResponseOutputTextDelta::new("r_1", "it_1", "")),
        "delta must be non-empty"
    );
    assert_eq!(
        encode_err(ResponseOutputTextDone::new("r_1", "in_1", "x")),
        "item_id must start with it_"
    );
    assert_eq!(encode_err(EndCall::new("")), "reason must be non-empty");
    assert_eq!(
        encode_err(SessionRejected::new("", true)),
        "code must be non-empty"
    );
    assert_eq!(
        encode_err(AgentError::new("e", "m").with_item_id("it_1")),
        "item_id requires response_id"
    );
    let mut bad_ts = SessionReady::new();
    bad_ts.ts = "yesterday".into();
    assert_eq!(encode_err(bad_ts), "ts must match the RFC 3339 profile");
    let mut bad_id = SessionReady::new();
    bad_id.id = String::new();
    assert_eq!(encode_err(bad_id), "id must be non-empty");
    let big = "x".repeat(MAX_FRAME_BYTES);
    assert_eq!(
        encode_err(ResponseOutputTextDone::new("r_1", "it_1", big)),
        "Voice message exceeds the maximum encoded frame size"
    );
}

#[test]
fn streaming_item_is_explicit_frames_and_validation_is_stateless() {
    let messages: Vec<OutboundVoiceMessage> = vec![
        ResponseCreated::reply("r_1", ["in_1"]).into(),
        ResponseOutputTextDelta::new("r_1", "it_1", "hel").into(),
        ResponseOutputTextDelta::new("r_1", "it_1", "lo").into(),
        ResponseOutputTextDone::new("r_1", "it_1", "hello").into(),
        ResponseDone::new("r_1").into(),
    ];
    let payloads: Vec<Value> = messages.into_iter().map(|m| encode(m).unwrap()).collect();
    assert_eq!(
        payloads
            .iter()
            .map(|p| p["type"].as_str().unwrap())
            .collect::<Vec<_>>(),
        [
            "response.created",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.done"
        ]
    );
    assert_eq!(payloads[3]["text"], "hello");
    assert_eq!(
        encode(ResponseDone::new("r_never_opened_here")).unwrap()["response_id"],
        "r_never_opened_here"
    );
}

#[test]
fn outbound_voice_alias_is_normalized_on_the_wire_only() {
    for (alias, normalized) in [
        ("azure-platform", "azure-standard"),
        ("custom", "azure-custom"),
    ] {
        let message = ResponseOutputTextDone::new("r_1", "it_1", "hello")
            .with_voice(object(json!({"type": alias})));
        assert_eq!(
            encode(message.clone()).unwrap()["voice"],
            json!({"type": normalized})
        );
        assert_eq!(message.voice, Some(object(json!({"type": alias}))));
    }
}

#[test]
fn outbound_voice_rejects_invalid_configuration() {
    let cases = [
        (
            json!({"type": "not-a-voice"}),
            "voice.type is not a supported Voice Live variant",
        ),
        (json!({"type": 1}), "voice.type must be a string"),
        (json!({}), "voice must be a non-empty mapping"),
        (json!({"name": null}), "voice.name must be a string"),
        (json!({"name": 42}), "voice.name must be a string"),
        (json!({"name": ""}), "voice.name must be non-empty"),
        (
            json!({"endpoint_id": null}),
            "voice.endpoint_id must be a string",
        ),
        (
            json!({"endpoint_id": 123}),
            "voice.endpoint_id must be a string",
        ),
        (
            json!({"endpoint_id": ""}),
            "voice.endpoint_id must be non-empty",
        ),
        (json!({"model": null}), "voice.model must be a string"),
        (json!({"model": []}), "voice.model must be a string"),
        (
            json!({"temperature": true}),
            "voice.temperature must be a number or null",
        ),
        (
            json!({"temperature": "0.5"}),
            "voice.temperature must be a number or null",
        ),
        (
            json!({"temperature": -0.01}),
            "voice.temperature must be between 0.0 and 1.0",
        ),
        (
            json!({"temperature": 1.01}),
            "voice.temperature must be between 0.0 and 1.0",
        ),
        (
            json!({"locale": 1}),
            "voice.locale must be a string or null",
        ),
        (
            json!({"style": false}),
            "voice.style must be a string or null",
        ),
        (json!({"pitch": []}), "voice.pitch must be a string or null"),
        (json!({"rate": {}}), "voice.rate must be a string or null"),
        (
            json!({"volume": 1.0}),
            "voice.volume must be a string or null",
        ),
        (
            json!({"custom_lexicon_url": 1}),
            "voice.custom_lexicon_url must be a string or null",
        ),
        (
            json!({"custom_text_normalization_url": []}),
            "voice.custom_text_normalization_url must be a string or null",
        ),
        (
            json!({"multi_talker_speaker_name": 1}),
            "voice.multi_talker_speaker_name must be a string or null",
        ),
        (
            json!({"prefer_locales": "en-US"}),
            "voice.prefer_locales must be an array of strings or null",
        ),
        (
            json!({"prefer_locales": ["en-US", 1]}),
            "voice.prefer_locales must be an array of strings or null",
        ),
    ];
    for (voice, expected) in cases {
        let message =
            ResponseOutputTextDone::new("r_1", "it_1", "hello").with_voice(object(voice.clone()));
        assert_eq!(encode_err(message.clone()), expected, "{voice}");
        // Repeatable, and the model is untouched.
        assert_eq!(encode_err(message.clone()), expected);
        assert_eq!(message.voice, Some(object(voice)));
    }
}

#[test]
fn outbound_voice_accepts_valid_fields_and_preserves_unknown_ones() {
    for temperature in [
        json!(0),
        json!(0.0),
        json!(0.5),
        json!(1),
        json!(1.0),
        json!(null),
    ] {
        let message = ResponseOutputTextDone::new("r_1", "it_1", "hello")
            .with_voice(object(json!({"temperature": temperature})));
        assert_eq!(
            encode(message).unwrap()["voice"],
            json!({"temperature": temperature})
        );
    }
    let message = ResponseOutputTextDone::new("r_1", "it_1", "hello")
        .with_voice(object(json!({"model": ""})));
    assert_eq!(encode(message).unwrap()["voice"], json!({"model": ""}));
    let voice = json!({
        "type": "azure-platform",
        "name": "en-US-Ava",
        "endpoint_id": "endpoint-1",
        "model": "model-1",
        "temperature": 0.5,
        "locale": null,
        "style": "chat",
        "pitch": null,
        "rate": "+10%",
        "volume": null,
        "custom_lexicon_url": "https://example.test/lexicon.xml",
        "custom_text_normalization_url": null,
        "prefer_locales": ["en-US", "fr-FR"],
        "multi_talker_speaker_name": null,
        "future_field": {"nested": [1, true, null]},
    });
    let message =
        ResponseOutputTextDelta::new("r_1", "it_1", "hello").with_voice(object(voice.clone()));
    let mut expected = voice;
    expected["type"] = json!("azure-standard");
    assert_eq!(encode(message).unwrap()["voice"], expected);
}

#[test]
fn debug_output_redacts_payloads_and_is_bounded() {
    let sensitive = "customer-secret-value";
    let ani = "+14255550123";
    let rendered = [
        format!(
            "{:?}",
            SessionStart {
                id: "m_1".into(),
                ts: TS.into(),
                protocol_version: "1.0".into(),
                reconnect: false,
                response_timeouts: ResponseTimeouts {
                    first_output_ms: 1,
                    idle_ms: 2,
                    max_duration_ms: 3
                },
                greeting: Some(sensitive.into()),
                no_input_timeout_ms: None,
                caller: Some(object(
                    json!({"ani": ani, "custom_parameters": {"secret": sensitive}})
                )),
            }
        ),
        format!(
            "{:?}",
            UserMessage {
                id: "m_2".into(),
                ts: TS.into(),
                item_id: "in_1".into(),
                content: vec![InputContentPart::InputText {
                    text: sensitive.into()
                }],
            }
        ),
        format!(
            "{:?}",
            SessionDisconnected {
                code: 1006,
                reason: Some(sensitive.into())
            }
        ),
        format!(
            "{:?}",
            SessionRejected::new("startup_failed", false).with_message(sensitive)
        ),
        format!(
            "{:?}",
            ResponseOutputTextDelta::new("r_1", "it_1", sensitive).with_voice(object(
                json!({"custom_lexicon_url": format!("https://example.test/?sig={sensitive}")})
            ))
        ),
        format!("{:?}", AgentError::new("backend_error", sensitive)),
        format!(
            "{:?}",
            InputContentPart::InputText {
                text: sensitive.into()
            }
        ),
    ];
    for text in rendered {
        assert!(text.len() <= 1024, "{text}");
        assert!(text.contains("<redacted>"), "{text}");
        assert!(!text.contains(sensitive) && !text.contains(ani), "{text}");
    }
    let shown = format!("{:?}", SessionRejected::new("startup_failed", false));
    assert!(
        shown.starts_with("SessionRejected(") && shown.contains("startup_failed"),
        "{shown}"
    );
    let timeout = ResponseTimeout {
        id: "m_1".into(),
        ts: TS.into(),
        stage: "first_output".into(),
        response_id: None,
        item_ids: Some(
            (0..100)
                .map(|i| format!("in_{i}_{}", "x".repeat(250)))
                .collect(),
        ),
    };
    let text = format!("{timeout:?}");
    assert!(text.starts_with("ResponseTimeout(") && text.contains(r#"stage="first_output""#));
    assert!(text.chars().count() <= 1024);
}

#[test]
fn generated_ids_and_timestamps_have_upstream_shapes() {
    for (id, prefix) in [
        (new_message_id(), "m_"),
        (new_response_id(), "r_"),
        (new_item_id(), "it_"),
    ] {
        let hex = id.strip_prefix(prefix).unwrap();
        assert_eq!(hex.len(), 32);
        assert!(hex
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()));
    }
    let ts = new_timestamp();
    assert_eq!(ts.len(), "2026-08-12T00:00:00.000Z".len());
    // It passes the codec's own timestamp profile.
    assert!(decode(
        &json!({"type": "session.end", "id": "m_1", "ts": ts, "reason": "x"}).to_string()
    )
    .is_ok());
}

// ---------------------------------------------------------------------------
// The host over a real WebSocket.

#[derive(Clone, Default)]
struct Recorder {
    log: Arc<Mutex<Vec<String>>>,
    sessions: Arc<Mutex<Vec<Session>>>,
}

impl Recorder {
    fn push(&self, entry: impl Into<String>) {
        self.log.lock().unwrap().push(entry.into());
    }

    fn entries(&self) -> Vec<String> {
        self.log.lock().unwrap().clone()
    }

    async fn wait_for(&self, entry: &str) {
        let found = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if self.entries().iter().any(|e| e == entry) {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await;
        assert!(found.is_ok(), "{entry} not in {:?}", self.entries());
    }
}

/// The sample's handshake, and a reply per user message keyed by its text.
struct Agent(Recorder);

#[async_trait::async_trait]
impl VoiceHandler for Agent {
    async fn on_session_start(&self, session: &Session, event: SessionStart) -> Result<()> {
        self.0.push(format!(
            "session.start:{}:{}",
            event.protocol_version, event.reconnect
        ));
        self.0.sessions.lock().unwrap().push(session.clone());
        if event.protocol_version != PROTOCOL_VERSION {
            session
                .send(SessionRejected::new("protocol_mismatch", false))
                .await?;
            return Ok(());
        }
        session.send(SessionReady::new()).await?;
        Ok(())
    }

    async fn on_user_message(&self, session: &Session, event: UserMessage) -> Result<()> {
        self.0.push(format!("user.message:{}", event.item_id));
        let text = match event.content.first() {
            Some(InputContentPart::InputText { text }) => text.clone(),
            Some(InputContentPart::InputData { data }) => {
                session
                    .send(ResponseCreated::reply("r_data", [&event.item_id]))
                    .await?;
                session
                    .send(ResponseOutputDataDone::new(
                        "r_data",
                        "it_data",
                        data.clone(),
                    ))
                    .await?;
                session.send(ResponseDone::new("r_data")).await?;
                return Ok(());
            }
            None => String::new(),
        };
        match text.as_str() {
            "fail" => return Err(Error::Other("callback failure".into())),
            "panic" => panic!("callback panic"),
            "silent" => {
                session
                    .send(ResponseNone::new([&event.item_id]).with_reason("acknowledged"))
                    .await?;
            }
            "goodbye" => {
                session
                    .send(EndCall::new("caller_done").with_mode(EndCallMode::Immediate))
                    .await?;
            }
            "error" => {
                session
                    .send(AgentError::new("backend_error", "failed").with_response_id("r_x"))
                    .await?;
            }
            "flood" => {
                // Output from an application task while the callback is busy.
                let sender = session.clone();
                tokio::spawn(async move {
                    for i in 0..200 {
                        let delta =
                            ResponseOutputTextDelta::new("r_flood", "it_flood", format!("{i} "));
                        if sender.send(delta).await.is_err() {
                            break;
                        }
                    }
                });
                tokio::time::sleep(Duration::from_millis(300)).await;
            }
            "until_gone" => {
                let sender = session.clone();
                let recorder = self.0.clone();
                tokio::spawn(async move {
                    loop {
                        let delta = ResponseOutputTextDelta::new("r_gone", "it_gone", "x");
                        if let Err(error) = sender.send(delta).await {
                            recorder.push(format!("send failed:{error}"));
                            return;
                        }
                        tokio::time::sleep(Duration::from_millis(5)).await;
                    }
                });
            }
            "turn" => {
                self.0.push(turn_checks(session).await);
            }
            _ => {
                let response_id = "r_reply";
                session
                    .send(ResponseCreated::reply(response_id, [&event.item_id]))
                    .await?;
                session
                    .send(ResponseOutputTextDelta::new(
                        response_id,
                        "it_reply",
                        "You said: ",
                    ))
                    .await?;
                session
                    .send(ResponseOutputTextDone::new(
                        response_id,
                        "it_reply",
                        format!("You said: {text}"),
                    ))
                    .await?;
                session.send(ResponseDone::new(response_id)).await?;
            }
        }
        Ok(())
    }

    async fn on_user_no_input(&self, _: &Session, event: UserNoInput) -> Result<()> {
        self.0
            .push(format!("user.no_input:{}:{}", event.item_id, event.count));
        Ok(())
    }

    async fn on_user_speech_started(&self, _: &Session, event: UserSpeechStarted) -> Result<()> {
        self.0.push(format!("user.speech_started:{}", event.id));
        Ok(())
    }

    async fn on_user_barge_in(&self, _: &Session, event: UserBargeIn) -> Result<()> {
        self.0.push(format!(
            "user.barge_in:{}:{:?}:{}",
            event.response_id, event.item_id, event.heard_text
        ));
        Ok(())
    }

    async fn on_response_accepted(&self, _: &Session, event: ResponseAccepted) -> Result<()> {
        self.0
            .push(format!("response.accepted:{}", event.response_id));
        Ok(())
    }

    async fn on_response_dropped(&self, _: &Session, event: ResponseDropped) -> Result<()> {
        self.0.push(format!(
            "response.dropped:{}:{}",
            event.response_id, event.reason
        ));
        Ok(())
    }

    async fn on_response_cancelled(&self, _: &Session, event: ResponseCancelled) -> Result<()> {
        self.0.push(format!(
            "response.cancelled:{}:{}",
            event.response_id, event.heard_text
        ));
        Ok(())
    }

    async fn on_response_timeout(&self, _: &Session, event: ResponseTimeout) -> Result<()> {
        self.0.push(format!(
            "response.timeout:{}:{:?}:{:?}",
            event.stage, event.response_id, event.item_ids
        ));
        Ok(())
    }

    async fn on_session_end(&self, session: &Session, event: SessionEnd) -> Result<()> {
        self.0.push(format!("session.end:{}", event.reason));
        if event.reason == "fail" {
            return Err(Error::Other("end failure".into()));
        }
        // Still writable inside the callback.
        session.send(SessionReady::new()).await?;
        Ok(())
    }

    async fn on_disconnect(&self, session: &Session, event: SessionDisconnected) -> Result<()> {
        self.0.push(format!(
            "disconnect:{}:{}",
            event.code,
            session.termination().map_or("none", |t| t.as_str())
        ));
        Ok(())
    }

    fn on_connection_terminating(&self, session: &Session) -> Result<()> {
        self.0.push(format!(
            "terminating:{}",
            session.termination().map_or("none", |t| t.as_str())
        ));
        Ok(())
    }
}

/// Target-turn rules, reported as one log line.
async fn turn_checks(session: &Session) -> String {
    let out = std::sync::Mutex::new(Vec::new());
    let note = |r: std::result::Result<(), VoiceError>| {
        out.lock().unwrap().push(match r {
            Ok(()) => "ok".to_string(),
            Err(e) => e.to_string(),
        })
    };
    note(
        session
            .start_target_turn(TargetTurnOrigin::Proactive, 1)
            .map(drop),
    );
    note(
        session
            .start_target_turn(TargetTurnOrigin::User, 0)
            .map(drop),
    );
    note(
        session
            .start_target_turn(TargetTurnOrigin::Proactive, 0)
            .map(drop),
    );
    note(TargetTurnOrigin::parse(" ").map(drop));
    let mut turn = session
        .start_target_turn(TargetTurnOrigin::User, 1)
        .unwrap();
    note(turn.activate(async {}).await);
    note(turn.activate(async {}).await);
    note(turn.complete(TargetTurnOutcome::Response, None, Some(1)));
    note(turn.complete(TargetTurnOutcome::None, Some("r_1"), Some(0)));
    note(turn.complete(TargetTurnOutcome::Error, None, Some(2)));
    note(turn.complete(TargetTurnOutcome::Response, Some("x_1"), Some(1)));
    note(turn.complete(TargetTurnOutcome::Response, Some("r_1"), Some(1)));
    note(turn.complete(TargetTurnOutcome::Response, None, None));
    out.lock().unwrap().push(turn.is_completed().to_string());
    note(turn.activate(async {}).await);
    out.into_inner().unwrap().join("|")
}

async fn serve(recorder: &Recorder) -> String {
    let host = VoiceAgentServerHost::new(Agent(recorder.clone()));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, host.into_router()).await.unwrap() });
    format!("ws://{addr}")
}

type Client =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

async fn connect(url: &str) -> (Client, tungstenite::http::Response<Option<Vec<u8>>>) {
    let request = url.into_client_request().unwrap();
    tokio_tungstenite::connect_async(request).await.unwrap()
}

async fn next(client: &mut Client) -> tungstenite::Message {
    loop {
        let message = tokio::time::timeout(Duration::from_secs(5), client.next())
            .await
            .expect("a frame")
            .expect("open")
            .unwrap();
        if !matches!(
            message,
            tungstenite::Message::Ping(_) | tungstenite::Message::Pong(_)
        ) {
            return message;
        }
    }
}

async fn next_json(client: &mut Client) -> Value {
    match next(client).await {
        tungstenite::Message::Text(text) => serde_json::from_str(text.as_str()).unwrap(),
        other => panic!("expected text, got {other:?}"),
    }
}

async fn close_of(client: &mut Client) -> (u16, String) {
    match next(client).await {
        tungstenite::Message::Close(Some(frame)) => (frame.code.into(), frame.reason.to_string()),
        other => panic!("expected a close frame, got {other:?}"),
    }
}

async fn send(client: &mut Client, text: impl Into<String>) {
    client
        .send(tungstenite::Message::text(text.into()))
        .await
        .unwrap();
}

#[tokio::test]
async fn both_endpoints_accept_the_handshake_and_relay_messages() {
    let recorder = Recorder::default();
    let base = serve(&recorder).await;
    for path in ["/invocations_ws", "/voice_conversation_engine"] {
        let (mut client, response) = connect(&format!("{base}{path}")).await;
        assert!(response
            .headers()
            .get("x-platform-server")
            .is_some_and(|v| v.to_str().unwrap().starts_with("agent-framework-rs/")));
        send(&mut client, session_start("1.0")).await;
        let ready = next_json(&mut client).await;
        assert_eq!(ready["type"], "session.ready");
        assert!(ready["id"].as_str().unwrap().starts_with("m_"));
        send(
            &mut client,
            frame("user.message", json!({"item_id": "in_question", "content": [{"type": "input_text", "text": "Hello"}]})),
        )
        .await;
        let created = next_json(&mut client).await;
        assert_eq!(created["type"], "response.created");
        assert_eq!(created["in_reply_to"], json!(["in_question"]));
        assert_eq!(next_json(&mut client).await["delta"], "You said: ");
        assert_eq!(next_json(&mut client).await["text"], "You said: Hello");
        assert_eq!(next_json(&mut client).await["type"], "response.done");
    }
}

#[tokio::test]
async fn the_handshake_rejects_another_protocol_version() {
    let recorder = Recorder::default();
    let base = serve(&recorder).await;
    let (mut client, _) = connect(&format!("{base}/voice_conversation_engine")).await;
    send(&mut client, session_start("2.0")).await;
    let rejected = next_json(&mut client).await;
    assert_eq!(rejected["type"], "session.rejected");
    assert_eq!(rejected["code"], "protocol_mismatch");
    assert_eq!(rejected["retriable"], false);
    assert!(rejected.get("message").is_none());
}

#[tokio::test]
async fn reconnect_is_relayed_and_connections_keep_separate_sessions() {
    let recorder = Recorder::default();
    let base = serve(&recorder).await;
    let (mut legacy, _) = connect(&format!("{base}/invocations_ws")).await;
    send(&mut legacy, session_start("1.0")).await;
    assert_eq!(next_json(&mut legacy).await["type"], "session.ready");
    {
        let (mut ce, _) = connect(&format!("{base}/voice_conversation_engine")).await;
        let mut start: Value = serde_json::from_str(&session_start("1.0")).unwrap();
        start["reconnect"] = json!(true);
        send(&mut ce, start.to_string()).await;
        assert_eq!(next_json(&mut ce).await["type"], "session.ready");
        ce.close(None).await.unwrap();
    }
    // Closing the CE connection leaves the legacy session working.
    send(
        &mut legacy,
        frame(
            "user.message",
            json!({"item_id": "in_legacy", "content": [{"type": "input_text", "text": "x"}]}),
        ),
    )
    .await;
    assert_eq!(next_json(&mut legacy).await["response_id"], "r_reply");
    let sessions = recorder.sessions.lock().unwrap().clone();
    assert_eq!(sessions.len(), 2);
    assert!(!sessions[0].same_connection(&sessions[1]));
    assert!(recorder
        .entries()
        .contains(&"session.start:1.0:true".to_string()));
}

#[tokio::test]
async fn the_superseded_route_is_not_served() {
    let recorder = Recorder::default();
    let base = serve(&recorder).await;
    let request = format!("{base}/conversation_engine")
        .into_client_request()
        .unwrap();
    assert!(tokio_tungstenite::connect_async(request).await.is_err());
}

#[tokio::test]
async fn every_inbound_event_reaches_its_callback() {
    let recorder = Recorder::default();
    let base = serve(&recorder).await;
    let (mut client, _) = connect(&format!("{base}/voice_conversation_engine")).await;
    send(&mut client, session_start("1.0")).await;
    assert_eq!(next_json(&mut client).await["type"], "session.ready");
    let frames = [
        frame(
            "user.message",
            json!({"item_id": "in_1", "content": [{"type": "input_text", "text": "silent"}]}),
        ),
        frame("user.no_input", json!({"item_id": "in_2", "count": 3})),
        frame("user.speech_started", json!({"id": "m_speech"})),
        // Only the typed barge-in is dispatched.
        frame("barge_in", json!({"response_id": "r_1", "heard_text": "x"})),
        frame(
            "user.barge_in",
            json!({"response_id": "r_1", "item_id": "it_1", "heard_text": "heard prefix"}),
        ),
        frame("response.accepted", json!({"response_id": "r_2"})),
        frame(
            "response.dropped",
            json!({"response_id": "r_3", "reason": "queue_full"}),
        ),
        frame(
            "response.cancelled",
            json!({"response_id": "r_4", "heard_text": "heard"}),
        ),
        frame(
            "response.timeout",
            json!({"item_ids": ["in_5", "in_6"], "stage": "first_output"}),
        ),
        frame(
            "response.timeout",
            json!({"response_id": "r_5", "stage": "idle"}),
        ),
        frame("session.end", json!({"reason": "caller_hangup"})),
        // After session.end nothing more is dispatched.
        frame("user.speech_started", json!({"id": "m_late"})),
    ];
    for text in frames {
        send(&mut client, text).await;
    }
    let none = next_json(&mut client).await;
    assert_eq!(none["type"], "response.none");
    assert_eq!(none["in_reply_to"], json!(["in_1"]));
    assert_eq!(none["reason"], "acknowledged");
    // The session.end callback may still send.
    assert_eq!(next_json(&mut client).await["type"], "session.ready");
    assert_eq!(close_of(&mut client).await, (1000, String::new()));
    recorder.wait_for("terminating:completed").await;
    assert_eq!(
        recorder.entries(),
        [
            "session.start:1.0:false",
            "user.message:in_1",
            "user.no_input:in_2:3",
            "user.speech_started:m_speech",
            "user.barge_in:r_1:Some(\"it_1\"):heard prefix",
            "response.accepted:r_2",
            "response.dropped:r_3:queue_full",
            "response.cancelled:r_4:heard",
            "response.timeout:first_output:None:Some([\"in_5\", \"in_6\"])",
            "response.timeout:idle:Some(\"r_5\"):None",
            "session.end:caller_hangup",
            "terminating:completed",
        ]
    );
    // The session is terminal: sends fail.
    let session = recorder.sessions.lock().unwrap()[0].clone();
    assert_eq!(session.termination(), Some(SessionTermination::Completed));
    assert_eq!(
        session.send(SessionReady::new()).await,
        Err(VoiceError::Terminating)
    );
    assert_eq!(
        session
            .send(SessionReady::new())
            .await
            .unwrap_err()
            .to_string(),
        "Voice Session is terminating"
    );
}

#[tokio::test]
async fn data_round_trips_and_end_call_and_error_encode() {
    let recorder = Recorder::default();
    let base = serve(&recorder).await;
    let (mut client, _) = connect(&format!("{base}/invocations_ws")).await;
    send(&mut client, session_start("1.0")).await;
    next_json(&mut client).await;
    let payload = json!({"type": "hubagent.metadata", "data": [true, null, {"key": "value"}]});
    send(
        &mut client,
        frame(
            "user.message",
            json!({"item_id": "in_data", "content": [{"type": "input_data", "data": payload}]}),
        ),
    )
    .await;
    assert_eq!(next_json(&mut client).await["type"], "response.created");
    assert_eq!(next_json(&mut client).await["data"], payload);
    assert_eq!(next_json(&mut client).await["type"], "response.done");

    send(
        &mut client,
        frame(
            "user.message",
            json!({"item_id": "in_e", "content": [{"type": "input_text", "text": "error"}]}),
        ),
    )
    .await;
    let error = next_json(&mut client).await;
    assert_eq!(
        (
            error["type"].as_str(),
            error["code"].as_str(),
            error["response_id"].as_str()
        ),
        (Some("error"), Some("backend_error"), Some("r_x"))
    );

    send(
        &mut client,
        frame(
            "user.message",
            json!({"item_id": "in_bye", "content": [{"type": "input_text", "text": "goodbye"}]}),
        ),
    )
    .await;
    let end_call = next_json(&mut client).await;
    assert_eq!(end_call["type"], "end_call");
    assert_eq!(end_call["reason"], "caller_done");
    assert_eq!(end_call["mode"], "immediate");
    // end_call asks the Bridge to end the call; the Bridge then sends
    // session.end, and the host closes normally.
    send(
        &mut client,
        frame("session.end", json!({"reason": "agent_ended"})),
    )
    .await;
    assert_eq!(next_json(&mut client).await["type"], "session.ready");
    assert_eq!(close_of(&mut client).await.0, 1000);
}

#[tokio::test]
async fn invalid_frames_close_with_upstream_codes_without_dispatch() {
    let recorder = Recorder::default();
    let base = serve(&recorder).await;
    let cases: Vec<(tungstenite::Message, u16)> = vec![
        (tungstenite::Message::text("not-json"), 1002),
        (tungstenite::Message::binary(b"binary".to_vec()), 1003),
        (
            tungstenite::Message::text("x".repeat(MAX_FRAME_BYTES + 1)),
            1009,
        ),
        (tungstenite::Message::text(frame("dtmf", json!({}))), 1003),
        (
            tungstenite::Message::text(frame("response.done", json!({"response_id": "r_1"}))),
            1002,
        ),
        (
            tungstenite::Message::text(frame(
                "user.message",
                json!({"item_id": "in_1", "content": [{"type": "input_image"}]}),
            )),
            1003,
        ),
    ];
    for (message, expected) in cases {
        let (mut client, _) = connect(&format!("{base}/invocations_ws")).await;
        client.send(message).await.unwrap();
        assert_eq!(close_of(&mut client).await, (expected, String::new()));
    }
    // Nothing was dispatched, and every connection ended as a protocol error.
    tokio::time::sleep(Duration::from_millis(100)).await;
    let entries = recorder.entries();
    assert_eq!(entries.len(), 6, "{entries:?}");
    assert!(
        entries.iter().all(|e| e == "terminating:protocol_error"),
        "{entries:?}"
    );
}

#[tokio::test]
async fn an_unknown_message_is_ignored_and_the_connection_continues() {
    let recorder = Recorder::default();
    let base = serve(&recorder).await;
    let (mut client, _) = connect(&format!("{base}/invocations_ws")).await;
    send(&mut client, frame("future.message", json!({"value": 1}))).await;
    send(&mut client, session_start("1.0")).await;
    assert_eq!(next_json(&mut client).await["type"], "session.ready");
}

#[tokio::test]
async fn failing_or_panicking_callbacks_close_with_an_internal_error() {
    for (text, termination) in [("fail", "callback_error"), ("panic", "callback_error")] {
        let recorder = Recorder::default();
        let base = serve(&recorder).await;
        let (mut client, _) = connect(&format!("{base}/invocations_ws")).await;
        send(
            &mut client,
            frame(
                "user.message",
                json!({"item_id": "in_1", "content": [{"type": "input_text", "text": text}]}),
            ),
        )
        .await;
        assert_eq!(
            close_of(&mut client).await,
            (1011, "Internal server error".to_string())
        );
        recorder
            .wait_for(&format!("terminating:{termination}"))
            .await;
    }
    // A failing session.end callback still ends dispatch, as a failure.
    let recorder = Recorder::default();
    let base = serve(&recorder).await;
    let (mut client, _) = connect(&format!("{base}/invocations_ws")).await;
    send(&mut client, frame("session.end", json!({"reason": "fail"}))).await;
    send(&mut client, frame("user.speech_started", json!({}))).await;
    assert_eq!(close_of(&mut client).await.0, 1011);
    recorder.wait_for("terminating:callback_error").await;
    assert!(!recorder
        .entries()
        .iter()
        .any(|e| e.starts_with("user.speech_started")));
}

#[tokio::test]
async fn a_peer_close_reaches_on_disconnect_with_its_code() {
    for (code, termination) in [
        (1000u16, "completed"),
        (4000, "transport_error"),
        (1008, "protocol_error"),
    ] {
        let recorder = Recorder::default();
        let base = serve(&recorder).await;
        let (mut client, _) = connect(&format!("{base}/voice_conversation_engine")).await;
        send(&mut client, session_start("1.0")).await;
        next_json(&mut client).await;
        client
            .close(Some(tungstenite::protocol::CloseFrame {
                code: code.into(),
                reason: "bye".into(),
            }))
            .await
            .unwrap();
        recorder
            .wait_for(&format!("disconnect:{code}:{termination}"))
            .await;
        assert!(recorder
            .entries()
            .contains(&format!("terminating:{termination}")));
        // Terminating runs before on_disconnect.
        let entries = recorder.entries();
        let t = entries
            .iter()
            .position(|e| e.starts_with("terminating"))
            .unwrap();
        let d = entries
            .iter()
            .position(|e| e.starts_with("disconnect"))
            .unwrap();
        assert!(t < d, "{entries:?}");
    }
}

#[tokio::test]
async fn a_busy_callback_does_not_stall_output() {
    let recorder = Recorder::default();
    let base = serve(&recorder).await;
    let (mut client, _) = connect(&format!("{base}/invocations_ws")).await;
    send(
        &mut client,
        frame(
            "user.message",
            json!({"item_id": "in_1", "content": [{"type": "input_text", "text": "flood"}]}),
        ),
    )
    .await;
    // The callback is still sleeping while all 200 deltas arrive.
    let received = tokio::time::timeout(Duration::from_millis(250), async {
        let mut count = 0;
        while count < 200 {
            if next_json(&mut client).await["type"] == "response.output_text.delta" {
                count += 1;
            }
        }
        count
    })
    .await;
    assert_eq!(received, Ok(200));
}

#[tokio::test]
async fn sends_fail_once_the_client_is_gone() {
    let recorder = Recorder::default();
    let base = serve(&recorder).await;
    let (mut client, _) = connect(&format!("{base}/invocations_ws")).await;
    send(
        &mut client,
        frame(
            "user.message",
            json!({"item_id": "in_1", "content": [{"type": "input_text", "text": "until_gone"}]}),
        ),
    )
    .await;
    assert_eq!(
        next_json(&mut client).await["type"],
        "response.output_text.delta"
    );
    client.close(None).await.unwrap();
    let failed = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if let Some(entry) = recorder
                .entries()
                .into_iter()
                .find(|e| e.starts_with("send failed:"))
            {
                return entry;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the application task's send fails");
    assert!(
        failed == "send failed:Voice Session is terminating"
            || failed.starts_with("send failed:Voice WebSocket send failed"),
        "{failed}"
    );
    // A close without a status code is 1005.
    recorder.wait_for("disconnect:1005:transport_error").await;
}

#[tokio::test]
async fn target_turns_validate_like_upstream() {
    let recorder = Recorder::default();
    let base = serve(&recorder).await;
    let (mut client, _) = connect(&format!("{base}/invocations_ws")).await;
    send(
        &mut client,
        frame(
            "user.message",
            json!({"item_id": "in_1", "content": [{"type": "input_text", "text": "turn"}]}),
        ),
    )
    .await;
    send(&mut client, frame("session.end", json!({"reason": "done"}))).await;
    next_json(&mut client).await;
    assert_eq!(close_of(&mut client).await.0, 1000);
    let max = i64::MAX;
    let expected = [
        "input_count must be zero for proactive target turns".to_string(),
        format!("input_count must be between 1 and {max}"),
        "ok".into(),
        "origin must not be empty".into(),
        "ok".into(),
        "Target turn has already been activated".into(),
        "response outcome requires a response_id and at least one output item".into(),
        "none outcome requires no response_id and zero output items".into(),
        "positive output_item_count requires a response_id".into(),
        "response_id must start with r_".into(),
        "ok".into(),
        // A completed turn ignores later completions.
        "ok".into(),
        "true".into(),
        "Target turn has already completed".into(),
    ]
    .join("|");
    assert!(
        recorder.entries().contains(&expected),
        "{:?}",
        recorder.entries()
    );
    // A terminal session starts no turn.
    let session = recorder.sessions.lock().unwrap().first().cloned();
    assert!(session.is_none(), "no session.start was sent");
}

#[test]
fn close_codes_classify_like_upstream() {
    assert_eq!(classify_close_code(1000), SessionTermination::Completed);
    assert_eq!(classify_close_code(1001), SessionTermination::Completed);
    for code in [1002, 1003, 1007, 1008, 1009, 1010] {
        assert_eq!(classify_close_code(code), SessionTermination::ProtocolError);
    }
    for code in [1005, 1006, 1011, 1015, 4000] {
        assert_eq!(
            classify_close_code(code),
            SessionTermination::TransportError
        );
    }
}
