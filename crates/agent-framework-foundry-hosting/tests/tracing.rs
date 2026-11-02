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
