//! The `x-ms-original-subject` platform header: parsing, model attribution
//! headers, and each surface's handling of a malformed value. Ported from
//! upstream's `tests/test_invoke.py`, `tests/test_ws_invoke.py` and
//! `tests/voice/test_voice_transport_findings.py` (invocations 1.3.0) and
//! core's `FoundryOriginalSubject` / `model_attribution_headers`.

mod common;

use std::sync::{Arc, Mutex};

use agent_framework_foundry_hosting::{
    FoundryOriginalSubject, FoundryRequestContext, ResponsesHostServer,
};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use common::*;
use serde_json::json;

const SUBJECT: &str = "x-ms-original-subject";

/// Upstream's malformed single-field values.
const MALFORMED: &[&str] = &[
    "malformed",
    r#"oid="one", oid="two""#,
    r#"unknown="value""#,
    r#"oid="one";parameter"#,
];

fn subject_headers(values: &[&str]) -> HeaderMap {
    let mut headers = HeaderMap::new();
    for value in values {
        headers.append(SUBJECT, HeaderValue::from_bytes(value.as_bytes()).unwrap());
    }
    headers
}

fn parse(values: &[&str]) -> Result<Option<FoundryOriginalSubject>, String> {
    FoundryOriginalSubject::from_headers(&subject_headers(values)).map_err(|e| e.to_string())
}

fn subject(oid: &str, tid: &str, idtype: &str) -> FoundryOriginalSubject {
    FoundryOriginalSubject {
        oid: Some(oid.into()),
        tid: Some(tid.into()),
        idtype: Some(idtype.into()),
    }
}

fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            out.push(u8::from_str_radix(&value[i + 1..i + 3], 16).unwrap());
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).unwrap()
}

/// The decoded subject member of an attribution baggage header.
fn baggage_subject(headers: &HeaderMap) -> String {
    let baggage = headers["x-ms-foundry-baggage"].to_str().unwrap();
    let first = baggage.split(',').next().unwrap();
    let (key, value) = first.split_once('=').unwrap();
    assert_eq!(key, "foundry.original_subject");
    percent_decode(value)
}

// ---------------------------------------------------------------------------
// Parsing.

#[test]
fn absent_header_is_none_and_valid_dictionaries_parse() {
    assert_eq!(parse(&[]), Ok(None));
    assert_eq!(
        parse(&[r#"oid="original-oid", tid="original-tid", idtype="app""#]),
        Ok(Some(subject("original-oid", "original-tid", "app")))
    );
    // Any order, optional whitespace, partial and empty members kept as sent.
    assert_eq!(
        parse(&["\tidtype=\"user\" ,\toid=\"\" "]),
        Ok(Some(FoundryOriginalSubject {
            oid: Some(String::new()),
            tid: None,
            idtype: Some("user".into()),
        }))
    );
    // Escaped quote and backslash are unescaped.
    assert_eq!(
        parse(&[r#"oid="a\"b\\c""#])
            .unwrap()
            .unwrap()
            .oid
            .as_deref(),
        Some(r#"a"b\c"#)
    );
    // Values are not printed.
    let debug = format!("{:?}", subject("secret-oid", "t", "user"));
    assert!(!debug.contains("secret-oid"), "{debug}");
}

#[test]
fn malformed_dictionaries_fail_with_upstream_messages() {
    let unique = "Original-subject attribution requires unique oid/tid/idtype string members.";
    let separated = "Original-subject attribution requires comma-separated dictionary members.";
    for (values, message) in [
        (&["malformed"][..], unique),
        (&[r#"oid="one", oid="two""#], unique),
        (&[r#"unknown="value""#], unique),
        (&[r#"OID="upper""#], unique),
        (&["oid=unquoted"], unique),
        (&[r#"oid="bad\escape""#], unique),
        (&["oid=\"tab\tinside\""], unique),
        (&[r#"oid="one";parameter"#], separated),
        (&[r#"oid="one","#], separated),
        (&[r#"oid="one" tid="two""#], separated),
        (&[r#"oid="one", "#], separated),
        (&[",oid=\"one\""], unique),
        (
            &[""],
            "Original-subject attribution requires a nonempty dictionary.",
        ),
        (
            &[" \t"],
            "Original-subject attribution requires a nonempty dictionary.",
        ),
        (
            &[r#"oid="one""#, r#"tid="two""#],
            "Original-subject attribution requires a single header field.",
        ),
    ] {
        assert_eq!(parse(values), Err(message.to_string()), "{values:?}");
    }
    for value in MALFORMED {
        assert!(parse(&[value]).is_err(), "{value}");
    }
    // A non-ASCII byte cannot match.
    let mut headers = HeaderMap::new();
    headers.insert(
        SUBJECT,
        HeaderValue::from_bytes(b"oid=\"caf\xe9\"").unwrap(),
    );
    assert!(FoundryOriginalSubject::from_headers(&headers).is_err());
}

#[test]
fn the_request_context_carries_the_subject_or_fails() {
    let mut headers = subject_headers(&[r#"oid="o", tid="t", idtype="user""#]);
    headers.insert("x-agent-user-id", "state-key".parse().unwrap());
    let context = FoundryRequestContext::from_headers(&headers, None).unwrap();
    assert_eq!(context.original_subject, Some(subject("o", "t", "user")));
    assert_eq!(context.user_id.as_deref(), Some("state-key"));
    assert_eq!(
        FoundryRequestContext::from_headers(&HeaderMap::new(), None)
            .unwrap()
            .original_subject,
        None
    );
    assert!(FoundryRequestContext::from_headers(&subject_headers(&["malformed"]), None).is_err());
}

// ---------------------------------------------------------------------------
// Model attribution headers.

#[test]
fn attribution_headers_carry_the_subject_in_baggage_only() {
    let raw = r#"oid="original-oid", tid="original-tid", idtype="app""#;
    let context = FoundryRequestContext {
        call_id: Some("call-1".into()),
        original_subject: Some(subject("original-oid", "original-tid", "app")),
        ..Default::default()
    };
    let mut existing = HeaderMap::new();
    existing.insert(SUBJECT, "oid=\"spoofed\"".parse().unwrap());
    existing.insert("x-ms-original-subject-oid", "spoofed".parse().unwrap());
    existing.insert("x-agent-foundry-call-id", "other".parse().unwrap());
    existing.insert("authorization", "Bearer model".parse().unwrap());
    existing.insert(
        "x-ms-foundry-baggage",
        "foundry.original_subject=old, application=shared,Foundry.Original_Subject.oid=x,\
         foundry%2Eoriginal_subject=y;p=1,,"
            .parse()
            .unwrap(),
    );
    let headers = context.model_attribution_headers(&existing).unwrap();
    assert_eq!(baggage_subject(&headers), raw);
    let baggage = headers["x-ms-foundry-baggage"].to_str().unwrap();
    assert_eq!(baggage.split(',').nth(1), Some("application=shared"));
    assert_eq!(baggage.split(',').count(), 2, "{baggage}");
    assert_eq!(headers["x-agent-foundry-call-id"], "call-1");
    assert_eq!(headers["authorization"], "Bearer model");
    assert!(!headers
        .keys()
        .any(|name| name.as_str().starts_with(SUBJECT)));
    assert_eq!(headers.len(), 3);
    // The input is not changed.
    assert_eq!(existing[SUBJECT], "oid=\"spoofed\"");
}

#[test]
fn attribution_headers_without_a_subject_keep_only_unrelated_baggage() {
    let context = FoundryRequestContext {
        call_id: Some("call-1".into()),
        ..Default::default()
    };
    assert!(context
        .model_attribution_headers(&HeaderMap::new())
        .unwrap()
        .is_empty());
    let mut existing = HeaderMap::new();
    existing.insert(
        "x-ms-foundry-baggage",
        "foundry.original_subject=old,application=shared"
            .parse()
            .unwrap(),
    );
    let headers = context.model_attribution_headers(&existing).unwrap();
    assert_eq!(headers.len(), 1);
    assert_eq!(headers["x-ms-foundry-baggage"], "application=shared");
}

#[test]
fn attribution_rejects_partial_unsafe_or_oversized_tuples() {
    let fails = |subject: FoundryOriginalSubject, baggage: Option<String>| {
        let context = FoundryRequestContext {
            original_subject: Some(subject),
            ..Default::default()
        };
        let mut headers = HeaderMap::new();
        if let Some(baggage) = baggage {
            headers.insert("x-ms-foundry-baggage", baggage.parse().unwrap());
        }
        context
            .model_attribution_headers(&headers)
            .unwrap_err()
            .to_string()
    };
    let unsafe_tuple = "Original-subject attribution requires a complete, HTTP-safe tuple.";
    let partial = FoundryOriginalSubject {
        oid: Some("o".into()),
        tid: Some("t".into()),
        idtype: None,
    };
    assert_eq!(fails(partial, None), unsafe_tuple);
    assert_eq!(fails(subject("", "t", "user"), None), unsafe_tuple);
    assert_eq!(fails(subject("has space", "t", "user"), None), unsafe_tuple);
    assert_eq!(fails(subject("caf\u{e9}", "t", "user"), None), unsafe_tuple);
    assert_eq!(
        fails(subject(&"o".repeat(240), "t", "user"), None),
        "Original-subject attribution requires a quoted tuple of at most 256 characters."
    );
    assert_eq!(
        fails(
            subject("o", "t", "user"),
            Some(format!("a={}", "x".repeat(1000)))
        ),
        "Original-subject attribution requires merged baggage of at most 1024 characters."
    );
    // Quotes and backslashes are escaped inside the quoted tuple.
    let context = FoundryRequestContext {
        original_subject: Some(subject(r#"a"b\c"#, "t", "user")),
        ..Default::default()
    };
    let headers = context
        .model_attribution_headers(&HeaderMap::new())
        .unwrap();
    assert_eq!(
        baggage_subject(&headers),
        r#"oid="a\"b\\c", tid="t", idtype="user""#
    );
}

// ---------------------------------------------------------------------------
// Responses: a malformed header is a platform-classified 400.

#[tokio::test]
async fn responses_reject_a_malformed_subject_before_the_agent_runs() {
    let echo = Echo::default();
    let router = ResponsesHostServer::agent(agent(echo.clone())).into_router();
    let mut request = post("/responses", json!({"input": "hi"}));
    request
        .headers_mut()
        .insert(SUBJECT, r#"oid="one", oid="two""#.parse().unwrap());
    let (status, headers, body) = send(&router, request).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let message = "Original-subject attribution requires unique oid/tid/idtype string members.";
    let body: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(body["error"]["message"], message);
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(body["error"]["code"], "invalid_request_error");
    assert_eq!(headers["x-platform-error-source"], "platform");
    assert_eq!(
        headers["x-platform-error-detail"],
        format!("ValueError: {message}").as_str()
    );
    assert!(echo.seen.lock().unwrap().is_empty());
}

#[tokio::test]
async fn responses_hand_the_subject_to_the_turn() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let recorded = seen.clone();
    let router = ResponsesHostServer::agent(agent(Echo::default()))
        .with_prepare_options(move |request, options| {
            recorded
                .lock()
                .unwrap()
                .push(request.context.original_subject.clone());
            Ok(options)
        })
        .into_router();
    let mut request = post("/responses", json!({"input": "hi"}));
    request.headers_mut().insert(
        SUBJECT,
        r#"oid="o", tid="t", idtype="user""#.parse().unwrap(),
    );
    let (status, _) = json(&router, request).await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = json(&router, post("/responses", json!({"input": "hi"}))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        *seen.lock().unwrap(),
        vec![Some(subject("o", "t", "user")), None]
    );
}

// ---------------------------------------------------------------------------
// Invocations (HTTP): 400 for invoke, get and cancel.

#[cfg(feature = "session-store")]
mod invocations {
    use super::*;
    use agent_framework_core::types::Message;
    use agent_framework_foundry_hosting::{InvocationRun, InvocationsHostServer};
    use axum::body::Body;
    use axum::http::Request;

    fn router(seen: Arc<Mutex<Vec<FoundryRequestContext>>>) -> axum::Router {
        InvocationsHostServer::agent(agent(Echo::default()))
            .with_parse_request(move |request| {
                seen.lock().unwrap().push(request.context.clone());
                Ok(InvocationRun {
                    messages: vec![Message::user("hi")],
                    stream: request.body["stream"].as_bool().unwrap_or(false),
                    ..Default::default()
                })
            })
            .into_router()
    }

    #[tokio::test]
    async fn a_malformed_subject_is_rejected_on_every_operation() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let router = router(seen.clone());
        let mut cases: Vec<Vec<&str>> = MALFORMED.iter().map(|v| vec![*v]).collect();
        cases.push(vec![r#"oid="one""#, r#"tid="two""#]);
        for values in &cases {
            for operation in ["invoke", "get", "cancel"] {
                let builder = match operation {
                    "invoke" => Request::post("/invocations?agent_session_id=s1")
                        .header("x-agent-invocation-id", "inv-1"),
                    "get" => Request::get("/invocations/inv-1"),
                    _ => Request::post("/invocations/inv-1/cancel"),
                };
                let mut request = builder.body(Body::from("{}")).unwrap();
                for (name, value) in &subject_headers(values) {
                    request.headers_mut().append(name, value.clone());
                }
                let (status, headers, body) = send(&router, request).await;
                assert_eq!(status, StatusCode::BAD_REQUEST, "{operation} {values:?}");
                assert_eq!(headers["x-platform-error-source"], "platform");
                assert_eq!(
                    headers["x-platform-error-detail"],
                    "invalid_original_subject"
                );
                assert_eq!(headers["x-agent-invocation-id"], "inv-1");
                if operation == "invoke" {
                    assert_eq!(headers["x-agent-session-id"], "s1");
                } else {
                    assert!(!headers.contains_key("x-agent-session-id"));
                }
                let body: serde_json::Value = serde_json::from_str(&body).unwrap();
                assert_eq!(
                    body,
                    json!({"error": {
                        "code": "invalid_request",
                        "message": "Invalid original-subject platform header.",
                    }})
                );
            }
        }
        assert!(seen.lock().unwrap().is_empty(), "no handler ran");
        // Without the header, get and cancel are unchanged.
        let (status, _, _) = send(&router, get("/invocations/inv-1")).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn the_subject_is_invocation_local() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let router = router(seen.clone());
        let raw = r#"oid="original-oid", tid="original-tid", idtype="app""#;
        for stream in [false, true] {
            let mut request = post(
                "/invocations?agent_session_id=s1",
                json!({"message": "hi", "stream": stream}),
            );
            request.headers_mut().insert(SUBJECT, raw.parse().unwrap());
            request
                .headers_mut()
                .insert("x-agent-user-id", "state-key".parse().unwrap());
            let (status, _, body) = send(&router, request).await;
            assert_eq!(status, StatusCode::OK, "{body}");
            let (status, _, body) = send(
                &router,
                post("/invocations?agent_session_id=s1", json!({"message": "hi"})),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{body}");
        }
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 4);
        for (i, context) in seen.iter().enumerate() {
            let forwarded = context
                .model_attribution_headers(&HeaderMap::new())
                .unwrap();
            if i % 2 == 0 {
                assert_eq!(
                    context.original_subject,
                    Some(subject("original-oid", "original-tid", "app"))
                );
                assert_eq!(baggage_subject(&forwarded), raw);
                assert_eq!(forwarded.len(), 1);
            } else {
                assert_eq!(context.original_subject, None);
                assert!(forwarded.is_empty());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The close event a WebSocket rejection logs.

#[cfg(feature = "websocket")]
mod capture {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use tracing::field::{Field, Visit};
    use tracing_subscriber::layer::{Context, SubscriberExt};
    use tracing_subscriber::Layer;

    /// Every event's fields, as strings.
    #[derive(Clone, Default)]
    pub struct Events(Arc<Mutex<Vec<HashMap<String, String>>>>);

    struct Fields(HashMap<String, String>);

    impl Visit for Fields {
        fn record_str(&mut self, field: &Field, value: &str) {
            self.0.insert(field.name().into(), value.into());
        }

        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            self.0.insert(field.name().into(), format!("{value:?}"));
        }
    }

    impl<S: tracing::Subscriber> Layer<S> for Events {
        fn on_event(&self, event: &tracing::Event<'_>, _: Context<'_, S>) {
            let mut fields = Fields(HashMap::new());
            event.record(&mut fields);
            self.0.lock().unwrap().push(fields.0);
        }
    }

    impl Events {
        /// Capture this thread's events (a current-thread runtime runs the
        /// server here too).
        pub fn install(&self) -> tracing::subscriber::DefaultGuard {
            tracing::subscriber::set_default(tracing_subscriber::registry().with(self.clone()))
        }

        /// Wait for an event whose `field` is `value`.
        pub async fn wait_for(&self, field: &str, value: &str) {
            for _ in 0..200 {
                if self
                    .0
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|e| e.get(field).is_some_and(|v| v == value))
                {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            panic!("no event with {field}={value}");
        }
    }
}

// ---------------------------------------------------------------------------
// `/invocations_ws`: accepted, then closed with 1011.

#[cfg(feature = "websocket")]
mod invocations_ws {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use agent_framework_foundry_hosting::invocations_ws::{InvocationsWs, Message};
    use futures::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::{self, client::IntoClientRequest};

    type Client = tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >;

    /// Replies to the first frame with the context's attribution headers.
    async fn serve(handled: Arc<AtomicUsize>) -> String {
        let ws = InvocationsWs::new(move |mut session| {
            let handled = handled.clone();
            async move {
                handled.fetch_add(1, Ordering::SeqCst);
                session.recv().await;
                let headers = session
                    .context()
                    .model_attribution_headers(&HeaderMap::new())
                    .unwrap();
                let forwarded: serde_json::Map<String, serde_json::Value> = headers
                    .iter()
                    .map(|(k, v)| (k.to_string(), json!(v.to_str().unwrap())))
                    .collect();
                session
                    .send(Message::Text(
                        serde_json::Value::Object(forwarded).to_string().into(),
                    ))
                    .await
            }
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, ws.into_router()).await.unwrap() });
        format!("ws://{addr}/invocations_ws")
    }

    async fn connect(url: &str, subject: Option<&str>) -> Client {
        let mut request = url.into_client_request().unwrap();
        if let Some(subject) = subject {
            request
                .headers_mut()
                .insert(SUBJECT, subject.parse().unwrap());
        }
        tokio_tungstenite::connect_async(request).await.unwrap().0
    }

    async fn next(client: &mut Client) -> tungstenite::Message {
        tokio::time::timeout(Duration::from_secs(5), client.next())
            .await
            .expect("a frame")
            .expect("open")
            .unwrap()
    }

    async fn forwarded(client: &mut Client) -> HeaderMap {
        client.send(tungstenite::Message::text("go")).await.unwrap();
        let tungstenite::Message::Text(text) = next(client).await else {
            panic!("expected text")
        };
        let value: serde_json::Value = serde_json::from_str(text.as_str()).unwrap();
        value
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.parse().unwrap(), v.as_str().unwrap().parse().unwrap()))
            .collect()
    }

    #[tokio::test]
    async fn a_malformed_subject_closes_with_1011_without_the_handler() {
        let events = capture::Events::default();
        let _guard = events.install();
        let handled = Arc::new(AtomicUsize::new(0));
        let url = serve(handled.clone()).await;
        for value in [MALFORMED[0], MALFORMED[1], MALFORMED[2]] {
            let mut client = connect(&url, Some(value)).await;
            match next(&mut client).await {
                tungstenite::Message::Close(Some(frame)) => {
                    assert_eq!(u16::from(frame.code), 1011);
                    assert_eq!(frame.reason.to_string(), "Internal server error");
                }
                other => panic!("expected a close frame, got {other:?}"),
            }
        }
        events
            .wait_for("error_code", "invalid_original_subject")
            .await;
        assert_eq!(handled.load(Ordering::SeqCst), 0);
        // A connection without the header still runs, without attribution.
        let mut client = connect(&url, None).await;
        assert!(forwarded(&mut client).await.is_empty());
    }

    #[tokio::test]
    async fn overlapping_connections_keep_their_own_subject() {
        let url = serve(Arc::default()).await;
        let alice = r#"oid="alice", tid="tenant-alice", idtype="user""#;
        let bob = r#"oid="bob", tid="tenant-bob", idtype="app""#;
        let mut first = connect(&url, Some(alice)).await;
        let mut second = connect(&url, Some(bob)).await;
        for (client, subject) in [(&mut first, alice), (&mut second, bob)] {
            let headers = forwarded(client).await;
            assert!(!headers.keys().any(|n| n.as_str().starts_with(SUBJECT)));
            assert_eq!(baggage_subject(&headers), subject);
            assert_eq!(headers.len(), 1);
        }
    }
}

// ---------------------------------------------------------------------------
// Voice: accepted, then closed with 1011 and no session or callback.

#[cfg(feature = "voice")]
mod voice {
    use super::*;
    use std::time::Duration;

    use agent_framework_core::error::Result;
    use agent_framework_foundry_hosting::voice::*;
    use futures::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::{self, client::IntoClientRequest};

    #[derive(Clone, Default)]
    struct Recorder(Arc<Mutex<Vec<String>>>);

    #[async_trait::async_trait]
    impl VoiceHandler for Recorder {
        async fn on_session_start(&self, session: &Session, _: SessionStart) -> Result<()> {
            let subject = session.context().original_subject.clone();
            self.0.lock().unwrap().push(format!(
                "session.start:{}",
                subject.and_then(|s| s.oid).unwrap_or_default()
            ));
            session.send(SessionReady::new()).await?;
            Ok(())
        }

        async fn on_disconnect(&self, _: &Session, _: SessionDisconnected) -> Result<()> {
            self.0.lock().unwrap().push("disconnect".into());
            Ok(())
        }

        fn on_connection_terminating(&self, _: &Session) -> Result<()> {
            self.0.lock().unwrap().push("terminating".into());
            Ok(())
        }
    }

    fn session_start() -> String {
        json!({
            "type": "session.start", "id": "m_1", "ts": "2026-08-12T00:00:00Z",
            "protocol_version": "1.0", "reconnect": false,
            "response_timeouts": {"first_output_ms": 1, "idle_ms": 2, "max_duration_ms": 3},
        })
        .to_string()
    }

    #[tokio::test]
    async fn a_malformed_subject_upgrades_then_closes_without_a_session() {
        let events = capture::Events::default();
        let _guard = events.install();
        let recorder = Recorder::default();
        let host = VoiceAgentServerHost::new(recorder.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, host.into_router()).await.unwrap() });
        for path in ["/invocations_ws", "/voice_conversation_engine"] {
            let mut request = format!("ws://{addr}{path}").into_client_request().unwrap();
            request
                .headers_mut()
                .insert(SUBJECT, "malformed".parse().unwrap());
            let (mut client, response) = tokio_tungstenite::connect_async(request).await.unwrap();
            assert_eq!(response.status(), 101, "accepted before the close");
            let _ = client
                .send(tungstenite::Message::text(session_start()))
                .await;
            let message = tokio::time::timeout(Duration::from_secs(5), client.next())
                .await
                .expect("a frame")
                .expect("open")
                .unwrap();
            match message {
                tungstenite::Message::Close(Some(frame)) => {
                    assert_eq!(u16::from(frame.code), 1011);
                    assert_eq!(frame.reason.to_string(), "Internal server error");
                }
                other => panic!("expected a close frame, got {other:?}"),
            }
        }
        events
            .wait_for(
                "azure.ai.agentserver.invocations_ws.error.code",
                "invalid_original_subject",
            )
            .await;
        assert!(recorder.0.lock().unwrap().is_empty(), "no callback ran");

        // A valid subject reaches the session's context.
        let mut request = format!("ws://{addr}/voice_conversation_engine")
            .into_client_request()
            .unwrap();
        request.headers_mut().insert(
            SUBJECT,
            r#"oid="voice-user", tid="tenant", idtype="user""#.parse().unwrap(),
        );
        let (mut client, _) = tokio_tungstenite::connect_async(request).await.unwrap();
        client
            .send(tungstenite::Message::text(session_start()))
            .await
            .unwrap();
        let ready = tokio::time::timeout(Duration::from_secs(5), client.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(ready.to_text().unwrap().contains("session.ready"));
        assert_eq!(
            recorder.0.lock().unwrap().as_slice(),
            ["session.start:voice-user"]
        );
    }
}
