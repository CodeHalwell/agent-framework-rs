//! Voice spans join the caller's trace (features `voice` and `otel`; its
//! own test binary: it installs a tracing subscriber). Ported from
//! upstream's `tests/voice/test_voice_tracing.py`.
#![cfg(all(feature = "voice", feature = "otel"))]

use std::time::Duration;

use agent_framework_core::error::Result;
use agent_framework_foundry_hosting::voice::*;
use agent_framework_foundry_hosting::FoundryHostConfig;
use futures::{SinkExt, StreamExt};
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider, SpanData};
use serde_json::json;
use tokio_tungstenite::tungstenite::{self, client::IntoClientRequest};
use tracing_subscriber::layer::SubscriberExt;

const TRACE_ID: &str = "11111111111111111111111111111111";
const PARENT_ID: &str = "2222222222222222";

struct Agent;

#[async_trait::async_trait]
impl VoiceHandler for Agent {
    async fn on_session_start(&self, session: &Session, _event: SessionStart) -> Result<()> {
        tracing::info_span!("customer.callback").in_scope(|| {});
        session.send(SessionReady::new()).await?;
        Ok(())
    }

    async fn on_user_message(&self, session: &Session, event: UserMessage) -> Result<()> {
        let response_id = new_response_id();
        let mut turn = session.start_target_turn(TargetTurnOrigin::User, 1)?;
        turn.activate(async {
            tracing::info_span!("customer.model").in_scope(|| {});
        })
        .await?;
        session
            .send(ResponseCreated::reply(&response_id, [event.item_id]))
            .await?;
        session
            .send(ResponseOutputTextDone::new(
                &response_id,
                new_item_id(),
                "hi",
            ))
            .await?;
        session.send(ResponseDone::new(&response_id)).await?;
        turn.complete(TargetTurnOutcome::Response, Some(&response_id), Some(1))?;
        Ok(())
    }

    async fn on_disconnect(&self, _session: &Session, _event: SessionDisconnected) -> Result<()> {
        tracing::info_span!("customer.disconnect").in_scope(|| {});
        Ok(())
    }
}

fn attr(span: &SpanData, key: &str) -> Option<String> {
    span.attributes
        .iter()
        .find(|kv| kv.key.as_str() == key)
        .map(|kv| kv.value.to_string())
}

async fn spans_named(
    exporter: &InMemorySpanExporter,
    provider: &SdkTracerProvider,
    name: &str,
) -> Vec<SpanData> {
    for _ in 0..200 {
        provider.force_flush().unwrap();
        let spans: Vec<SpanData> = exporter
            .get_finished_spans()
            .unwrap()
            .into_iter()
            .filter(|s| s.name == name)
            .collect();
        if !spans.is_empty() {
            return spans;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("no {name} span");
}

type Client =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

async fn next_text(client: &mut Client) -> serde_json::Value {
    loop {
        match client.next().await.unwrap().unwrap() {
            tungstenite::Message::Text(text) => {
                return serde_json::from_str(text.as_str()).unwrap()
            }
            tungstenite::Message::Close(_) => panic!("closed"),
            _ => {}
        }
    }
}

fn frame(message_type: &str, fields: serde_json::Value) -> tungstenite::Message {
    let mut payload = json!({"type": message_type, "id": "m_1", "ts": "2026-08-17T00:00:00Z"});
    payload
        .as_object_mut()
        .unwrap()
        .extend(fields.as_object().unwrap().clone());
    tungstenite::Message::text(payload.to_string())
}

fn session_start(version: &str) -> tungstenite::Message {
    frame(
        "session.start",
        json!({
            "protocol_version": version,
            "reconnect": false,
            "response_timeouts": {"first_output_ms": 1000, "idle_ms": 1000, "max_duration_ms": 3000},
        }),
    )
}

async fn connect(traceparent: bool) -> Client {
    let config = FoundryHostConfig::from_lookup(|name| {
        (name == "FOUNDRY_AGENT_SESSION_ID").then(|| "session_safe_123".to_string())
    })
    .unwrap();
    let host = VoiceAgentServerHost::new(Agent).with_config(config);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, host.into_router()).await.unwrap() });
    let mut request = format!("ws://{addr}/invocations_ws")
        .into_client_request()
        .unwrap();
    if traceparent {
        request.headers_mut().insert(
            "traceparent",
            format!("00-{TRACE_ID}-{PARENT_ID}-01").parse().unwrap(),
        );
    }
    tokio_tungstenite::connect_async(request).await.unwrap().0
}

fn install() -> (
    InMemorySpanExporter,
    SdkTracerProvider,
    tracing::subscriber::DefaultGuard,
) {
    let exporter = InMemorySpanExporter::default();
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(exporter.clone())
        .build();
    let subscriber = tracing_subscriber::registry()
        .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("test")));
    let guard = tracing::subscriber::set_default(subscriber);
    (exporter, provider, guard)
}

#[tokio::test]
async fn connection_callback_and_target_spans_have_explicit_parents() {
    let (exporter, provider, _guard) = install();
    let mut client = connect(true).await;
    client.send(session_start("1.0")).await.unwrap();
    assert_eq!(next_text(&mut client).await["type"], "session.ready");
    client
        .send(frame(
            "user.message",
            json!({"item_id": "in_1", "content": [{"type": "input_text", "text": "hi"}]}),
        ))
        .await
        .unwrap();
    assert_eq!(next_text(&mut client).await["type"], "response.created");
    next_text(&mut client).await;
    let done = next_text(&mut client).await;
    let response_id = done["response_id"].as_str().unwrap().to_string();
    client
        .send(frame("session.end", json!({"reason": "completed"})))
        .await
        .unwrap();

    let connection = spans_named(&exporter, &provider, "agentserver.connection")
        .await
        .remove(0);
    assert_eq!(connection.span_context.trace_id().to_string(), TRACE_ID);
    assert_eq!(connection.parent_span_id.to_string(), PARENT_ID);
    assert_eq!(connection.span_kind, opentelemetry::trace::SpanKind::Server);
    assert_eq!(
        attr(&connection, "bridge.outcome").as_deref(),
        Some("completed")
    );
    assert_eq!(
        attr(
            &connection,
            "azure.ai.agentserver.invocations_ws.close_code"
        )
        .as_deref(),
        Some("1000")
    );
    assert_eq!(
        attr(
            &connection,
            "azure.ai.agentserver.invocations_ws.session_id"
        )
        .as_deref(),
        Some("session_safe_123")
    );
    assert_eq!(
        attr(
            &connection,
            "azure.ai.agentserver.invocations_ws.protocol_version"
        )
        .as_deref(),
        Some("1.0")
    );
    assert_eq!(
        attr(&connection, "azure.ai.agentserver.invocations_ws.reconnect").as_deref(),
        Some("false")
    );
    assert_eq!(
        attr(&connection, "network.protocol.name").as_deref(),
        Some("websocket")
    );
    assert_eq!(attr(&connection, "error.type"), None);

    let callbacks = spans_named(&exporter, &provider, "voice.callback").await;
    let start = callbacks
        .iter()
        .find(|s| attr(s, "voice.event.type").as_deref() == Some("session.start"))
        .expect("a session.start callback span");
    assert_eq!(start.parent_span_id, connection.span_context.span_id());
    let customer = spans_named(&exporter, &provider, "customer.callback")
        .await
        .remove(0);
    assert_eq!(customer.parent_span_id, start.span_context.span_id());

    let turn = spans_named(&exporter, &provider, "invoke_agent")
        .await
        .remove(0);
    assert_eq!(turn.parent_span_id, connection.span_context.span_id());
    assert_eq!(
        attr(&turn, "gen_ai.operation.name").as_deref(),
        Some("invoke_agent")
    );
    assert_eq!(attr(&turn, "turn.origin").as_deref(), Some("user"));
    assert_eq!(attr(&turn, "bridge.input.count").as_deref(), Some("1"));
    assert_eq!(attr(&turn, "bridge.outcome").as_deref(), Some("response"));
    assert_eq!(
        attr(&turn, "bridge.output.item_count").as_deref(),
        Some("1")
    );
    assert_eq!(attr(&turn, "gen_ai.response.id"), Some(response_id));
    let model = spans_named(&exporter, &provider, "customer.model")
        .await
        .remove(0);
    assert_eq!(model.parent_span_id, turn.span_context.span_id());
}

#[tokio::test]
async fn unsafe_protocol_versions_stay_off_the_span_and_disconnects_are_children() {
    let (exporter, provider, _guard) = install();
    let mut client = connect(false).await;
    client
        .send(session_start("private-protocol-sentinel"))
        .await
        .unwrap();
    assert_eq!(next_text(&mut client).await["type"], "session.ready");
    client.close(None).await.unwrap();

    let connection = spans_named(&exporter, &provider, "agentserver.connection")
        .await
        .remove(0);
    assert_eq!(
        attr(
            &connection,
            "azure.ai.agentserver.invocations_ws.protocol_version"
        ),
        None
    );
    // No traceparent: a new root.
    assert_eq!(
        connection.parent_span_id,
        opentelemetry::trace::SpanId::INVALID
    );
    // A close without a status code (1005) is a transport error.
    assert_eq!(
        attr(&connection, "bridge.outcome").as_deref(),
        Some("transport_error")
    );
    assert_eq!(
        attr(&connection, "error.type").as_deref(),
        Some("transport_error")
    );
    let disconnect = spans_named(&exporter, &provider, "voice.callback")
        .await
        .into_iter()
        .find(|s| attr(s, "voice.event.type").as_deref() == Some("disconnect"))
        .expect("a disconnect callback span");
    assert_eq!(disconnect.parent_span_id, connection.span_context.span_id());
    let customer = spans_named(&exporter, &provider, "customer.disconnect")
        .await
        .remove(0);
    assert_eq!(customer.parent_span_id, disconnect.span_context.span_id());
}
