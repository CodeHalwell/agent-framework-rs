//! GenAI metrics via the `otel-metrics` feature: `ObservableChatClient`
//! records two histograms through the `opentelemetry` API --
//! `gen_ai.client.token.usage` and `gen_ai.client.operation.duration` -- into
//! whichever `MeterProvider` is installed process-wide via
//! `opentelemetry::global::set_meter_provider`.
//!
//! This example installs an in-memory `opentelemetry_sdk` `MeterProvider`
//! (an `InMemoryMetricExporter` behind a `PeriodicReader`) as a stand-in for
//! a real OTLP exporter, runs a few calls through `ObservableChatClient` over
//! a canned client, then flushes and prints what was recorded. Runs fully
//! offline -- no API key or network needed.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example otel_metrics
//! ```

use agent_framework::observability::metrics::{OPERATION_DURATION_METRIC, TOKEN_USAGE_METRIC};
use agent_framework::observability::ObservableChatClient;
use agent_framework::prelude::*;
use async_trait::async_trait;
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, Metric, MetricData, ResourceMetrics};
use opentelemetry_sdk::metrics::{
    InMemoryMetricExporterBuilder, PeriodicReader, SdkMeterProvider, Temporality,
};

/// A minimal offline stand-in for a model, with fixed token usage so the
/// recorded histograms have predictable values.
#[derive(Clone, Default)]
struct CannedClient;

#[async_trait]
impl ChatClient for CannedClient {
    async fn get_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        let mut resp = ChatResponse::from_text("The answer is 42.");
        resp.model = Some("canned-model-v1".to_string());
        resp.finish_reason = Some(FinishReason::stop());
        resp.usage_details = Some(UsageDetails {
            input_token_count: Some(12),
            output_token_count: Some(6),
            total_token_count: Some(18),
            ..Default::default()
        });
        Ok(resp)
    }

    async fn get_streaming_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatStream> {
        Ok(Box::pin(futures::stream::empty()))
    }

    fn model(&self) -> Option<&str> {
        Some("canned-model-v1")
    }
}

/// Find a metric by name across every resource/scope, mirroring the helper
/// the core crate's own `otel-metrics` tests use.
fn find_metric<'a>(resource_metrics: &'a [ResourceMetrics], name: &str) -> Option<&'a Metric> {
    resource_metrics
        .iter()
        .flat_map(|rm| rm.scope_metrics())
        .flat_map(|sm| sm.metrics())
        .find(|m| m.name() == name)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Install an in-memory MeterProvider as the process-global provider --
    // in production, swap this reader for one wrapping a real OTLP exporter
    // (see the `observability` module docs for the exact snippet).
    let exporter = InMemoryMetricExporterBuilder::new()
        .with_temporality(Temporality::Delta)
        .build();
    let reader = PeriodicReader::builder(exporter.clone()).build();
    let provider = SdkMeterProvider::builder().with_reader(reader).build();
    opentelemetry::global::set_meter_provider(provider.clone());

    // Run a few chat calls through the decorator. Each records into the two
    // GenAI histograms above (token usage split by input/output type, plus
    // operation duration) -- purely via the `opentelemetry` API crate; no
    // SDK dependency exists inside `agent-framework-core` itself.
    let client = ObservableChatClient::new(CannedClient, "demo-provider");
    for _ in 0..3 {
        let _ = client
            .get_response(
                vec![Message::user("What is the answer?")],
                ChatOptions::new().with_model("demo-model"),
            )
            .await?;
    }

    // Flush the periodic reader and read back what was exported.
    provider
        .force_flush()
        .map_err(|e| Error::other(e.to_string()))?;
    let finished = exporter
        .get_finished_metrics()
        .map_err(|e| Error::other(e.to_string()))?;

    println!("-- recorded metrics after 3 calls --\n");

    if let Some(metric) = find_metric(&finished, TOKEN_USAGE_METRIC) {
        println!("{} ({})", metric.name(), metric.unit());
        if let AggregatedMetrics::U64(MetricData::Histogram(hist)) = metric.data() {
            for point in hist.data_points() {
                let token_type = point
                    .attributes()
                    .find(|kv| kv.key.as_str() == "gen_ai.token.type")
                    .map(|kv| kv.value.to_string())
                    .unwrap_or_default();
                println!(
                    "  token_type={token_type:<8} count={:<3} sum={}",
                    point.count(),
                    point.sum()
                );
            }
        }
    }
    println!();

    if let Some(metric) = find_metric(&finished, OPERATION_DURATION_METRIC) {
        println!("{} ({})", metric.name(), metric.unit());
        if let AggregatedMetrics::F64(MetricData::Histogram(hist)) = metric.data() {
            for point in hist.data_points() {
                println!("  count={} sum={:.6}s", point.count(), point.sum());
            }
        }
    }

    Ok(())
}
