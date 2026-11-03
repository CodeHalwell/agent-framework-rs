#![cfg(feature = "otel")]
//! Request spans join the caller's W3C trace (its own test binary: it
//! installs a tracing subscriber).

mod common;

use agent_framework_foundry_hosting::ResponsesHostServer;
use axum::http::StatusCode;
use common::*;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider};
use tracing_subscriber::layer::SubscriberExt;

#[tokio::test]
async fn request_spans_are_children_of_the_callers_trace() {
    let exporter = InMemorySpanExporter::default();
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(exporter.clone())
        .build();
    let subscriber = tracing_subscriber::registry()
        .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("test")));
    let _guard = tracing::subscriber::set_default(subscriber);

    let router = ResponsesHostServer::agent(agent(Echo::default())).into_router();
    let mut request = get("/readiness");
    request.headers_mut().insert(
        "traceparent",
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
            .parse()
            .unwrap(),
    );
    let (status, _, _) = send(&router, request).await;
    assert_eq!(status, StatusCode::OK);
    provider.force_flush().unwrap();
    let spans = exporter.get_finished_spans().unwrap();
    let span = spans
        .iter()
        .find(|s| s.name == "foundry.request")
        .expect("a request span");
    assert_eq!(
        span.span_context.trace_id().to_string(),
        "4bf92f3577b34da6a3ce929d0e0e4736"
    );
    assert_eq!(span.parent_span_id.to_string(), "00f067aa0ba902b7");
}

#[tokio::test]
async fn foundry_enrichment_stamps_every_span() {
    use agent_framework_foundry_hosting::{FoundryEnrichmentSpanProcessor, FoundryHostConfig};
    let config = FoundryHostConfig::from_lookup(|name| {
        match name {
            "FOUNDRY_AGENT_NAME" => Some("weather"),
            "FOUNDRY_AGENT_VERSION" => Some("3"),
            "FOUNDRY_PROJECT_ARM_ID" => Some("/subscriptions/s/projects/p"),
            "FOUNDRY_AGENT_SESSION_ID" => Some("hosted-session"),
            _ => None,
        }
        .map(str::to_string)
    })
    .unwrap();
    let exporter = InMemorySpanExporter::default();
    let provider = SdkTracerProvider::builder()
        .with_span_processor(FoundryEnrichmentSpanProcessor::from_config(&config))
        .with_simple_exporter(exporter.clone())
        .build();
    let subscriber = tracing_subscriber::registry()
        .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("test")));
    let _guard = tracing::subscriber::set_default(subscriber);

    let router = ResponsesHostServer::agent(agent(Echo::default()))
        .with_config(config.clone())
        .into_router();
    let mut request = get("/readiness");
    request.headers_mut().insert(
        "baggage",
        "azure.ai.agentserver.conversation_id=conv_9,azure.ai.agentserver.invocation_id=inv_2"
            .parse()
            .unwrap(),
    );
    let (status, _, _) = send(&router, request).await;
    assert_eq!(status, StatusCode::OK);
    // A span of the application's own, outside any request.
    tracing::info_span!("app.work").in_scope(|| {});
    provider.force_flush().unwrap();
    let spans = exporter.get_finished_spans().unwrap();
    let attr = |span: &opentelemetry_sdk::trace::SpanData, key: &str| {
        span.attributes
            .iter()
            .find(|kv| kv.key.as_str() == key)
            .map(|kv| kv.value.to_string())
    };
    for span in &spans {
        assert_eq!(
            attr(span, "microsoft.foundry").as_deref(),
            Some("True"),
            "{}",
            span.name
        );
        assert_eq!(attr(span, "gen_ai.agent.name").as_deref(), Some("weather"));
        assert_eq!(attr(span, "gen_ai.agent.version").as_deref(), Some("3"));
        assert_eq!(attr(span, "gen_ai.agent.id").as_deref(), Some("weather:3"));
        assert_eq!(
            attr(span, "microsoft.foundry.project.id").as_deref(),
            Some("/subscriptions/s/projects/p")
        );
        assert_eq!(
            attr(span, "microsoft.session.id").as_deref(),
            Some("hosted-session")
        );
    }
    let request = spans
        .iter()
        .find(|s| s.name == "foundry.request")
        .expect("a request span");
    assert_eq!(
        attr(request, "gen_ai.conversation.id").as_deref(),
        Some("conv_9")
    );
    assert_eq!(
        attr(request, "azure.ai.agentserver.invocations.invocation_id").as_deref(),
        Some("inv_2")
    );
    let app = spans
        .iter()
        .find(|s| s.name == "app.work")
        .expect("the app span");
    assert_eq!(attr(app, "gen_ai.conversation.id"), None);
}
