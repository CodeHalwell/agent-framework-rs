//! In-memory metrics assertions for the `otel-metrics` feature.
//!
//! Run with `cargo test -p agent-framework-core --features otel-metrics`;
//! this whole file is compiled out otherwise (see the `#![cfg]` below), so it
//! contributes nothing to a default build/test run.
//!
//! Like `tests/observability.rs`, this is a dedicated test binary, and for a
//! similar reason: [`opentelemetry::global`]'s meter provider is
//! process-global, and [`opentelemetry::global::meter`] binds a `Meter` to
//! whichever provider is installed *at the time of that call* — a later
//! `set_meter_provider` does not retroactively affect `Meter`s (or, in this
//! crate, the lazily-`OnceLock`-cached instruments built from one) already
//! obtained. All tests in this binary therefore share one process-wide
//! `SdkMeterProvider` + `InMemoryMetricExporter` (installed once, lazily, by
//! [`harness`]) and are serialized by `METRICS_TEST_MUTEX` so each test can
//! `reset()` the exporter beforehand and only see its own recordings.

#![cfg(feature = "otel-metrics")]

use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use agent_framework_core::observability::metrics::{
    self, FUNCTION_INVOCATION_DURATION_METRIC, OPERATION_DURATION_BUCKET_BOUNDARIES,
    OPERATION_DURATION_METRIC, TOKEN_USAGE_BUCKET_BOUNDARIES, TOKEN_USAGE_METRIC,
};
use agent_framework_core::observability::ObservableChatClient;
use agent_framework_core::prelude::*;
use async_trait::async_trait;
use opentelemetry_sdk::metrics::{
    InMemoryMetricExporter, InMemoryMetricExporterBuilder, PeriodicReader, SdkMeterProvider,
    Temporality,
};

#[derive(Clone, Default)]
struct StubClient {
    fail: bool,
}

#[async_trait]
impl ChatClient for StubClient {
    async fn get_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        if self.fail {
            return Err(Error::service("boom"));
        }
        let mut resp = ChatResponse::from_text("hi");
        resp.finish_reason = Some(FinishReason::stop());
        resp.model = Some("resolved-model".to_string());
        resp.usage_details = Some(UsageDetails {
            input_token_count: Some(11),
            output_token_count: Some(4),
            total_token_count: Some(15),
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
        Some("stub-model")
    }
}

static METRICS_TEST_MUTEX: Mutex<()> = Mutex::new(());
static HARNESS: OnceLock<(SdkMeterProvider, InMemoryMetricExporter)> = OnceLock::new();

/// Install (once, lazily) a process-wide `SdkMeterProvider` reading into an
/// in-memory exporter, and return handles to both. Must only be called while
/// holding `METRICS_TEST_MUTEX`.
///
/// Uses `Temporality::Delta` rather than the default `Cumulative`:
/// cumulative temporality re-reports every previously-recorded data point's
/// running total on *every* collection, so `exporter.reset()` (which only
/// clears the exporter's own buffer, not the provider's aggregation state)
/// would not give later tests a clean slate — a later test with no
/// recordings of its own would still see the earlier tests' data points.
/// Delta temporality reports only what changed since the last collection, so
/// each test's `force_flush()` + `reset()` genuinely isolates it.
fn harness() -> &'static (SdkMeterProvider, InMemoryMetricExporter) {
    HARNESS.get_or_init(|| {
        let exporter = InMemoryMetricExporterBuilder::new()
            .with_temporality(Temporality::Delta)
            .build();
        let reader = PeriodicReader::builder(exporter.clone()).build();
        let provider = SdkMeterProvider::builder().with_reader(reader).build();
        opentelemetry::global::set_meter_provider(provider.clone());
        (provider, exporter)
    })
}

fn find_metric<'a>(
    resource_metrics: &'a [opentelemetry_sdk::metrics::data::ResourceMetrics],
    name: &str,
) -> Option<&'a opentelemetry_sdk::metrics::data::Metric> {
    resource_metrics
        .iter()
        .flat_map(|rm| rm.scope_metrics())
        .flat_map(|sm| sm.metrics())
        .find(|m| m.name() == name)
}

fn attr_value(kvs: &[opentelemetry::KeyValue], key: &str) -> Option<String> {
    kvs.iter()
        .find(|kv| kv.key.as_str() == key)
        .map(|kv| kv.value.to_string())
}

#[test]
fn chat_completion_records_token_usage_and_operation_duration() {
    let _guard = METRICS_TEST_MUTEX.lock().unwrap();
    let (provider, exporter) = harness();
    exporter.reset();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let client = ObservableChatClient::new(StubClient::default(), "stub-provider");
        let _ = client
            .get_response(
                vec![Message::user("hi")],
                ChatOptions::new().with_model("request-model"),
            )
            .await
            .unwrap();
    });

    provider.force_flush().unwrap();
    let finished = exporter.get_finished_metrics().unwrap();

    // -- gen_ai.client.token.usage ---------------------------------
    let token_usage = find_metric(&finished, TOKEN_USAGE_METRIC)
        .unwrap_or_else(|| panic!("{TOKEN_USAGE_METRIC} not recorded"));
    assert_eq!(token_usage.unit(), "tokens");
    let opentelemetry_sdk::metrics::data::AggregatedMetrics::U64(
        opentelemetry_sdk::metrics::data::MetricData::Histogram(hist),
    ) = token_usage.data()
    else {
        panic!(
            "expected a u64 histogram for {TOKEN_USAGE_METRIC}, got {:?}",
            token_usage.data()
        );
    };
    let points: Vec<_> = hist.data_points().collect();
    assert_eq!(points.len(), 2, "expected one data point per token type");

    let input_point = points
        .iter()
        .find(|p| {
            p.attributes()
                .any(|kv| kv.key.as_str() == "gen_ai.token.type" && kv.value.to_string() == "input")
        })
        .expect("input token-usage data point");
    assert_eq!(input_point.count(), 1);
    assert_eq!(input_point.sum(), 11);
    let input_attrs: Vec<_> = input_point.attributes().cloned().collect();
    assert_eq!(
        attr_value(&input_attrs, "gen_ai.operation.name").as_deref(),
        Some("chat")
    );
    assert_eq!(
        attr_value(&input_attrs, "gen_ai.provider.name").as_deref(),
        Some("stub-provider")
    );
    assert_eq!(
        attr_value(&input_attrs, "gen_ai.request.model").as_deref(),
        Some("request-model")
    );
    assert_eq!(
        attr_value(&input_attrs, "gen_ai.response.model").as_deref(),
        Some("resolved-model")
    );

    let output_point = points
        .iter()
        .find(|p| {
            p.attributes().any(|kv| {
                kv.key.as_str() == "gen_ai.token.type" && kv.value.to_string() == "output"
            })
        })
        .expect("output token-usage data point");
    assert_eq!(output_point.count(), 1);
    assert_eq!(output_point.sum(), 4);

    assert_eq!(
        hist.data_points()
            .next()
            .unwrap()
            .bounds()
            .collect::<Vec<_>>(),
        TOKEN_USAGE_BUCKET_BOUNDARIES.to_vec()
    );

    // -- gen_ai.client.operation.duration ---------------------------
    let duration = find_metric(&finished, OPERATION_DURATION_METRIC)
        .unwrap_or_else(|| panic!("{OPERATION_DURATION_METRIC} not recorded"));
    assert_eq!(duration.unit(), "s");
    let opentelemetry_sdk::metrics::data::AggregatedMetrics::F64(
        opentelemetry_sdk::metrics::data::MetricData::Histogram(hist),
    ) = duration.data()
    else {
        panic!(
            "expected an f64 histogram for {OPERATION_DURATION_METRIC}, got {:?}",
            duration.data()
        );
    };
    let points: Vec<_> = hist.data_points().collect();
    assert_eq!(points.len(), 1);
    assert_eq!(points[0].count(), 1);
    assert!(points[0].sum() >= 0.0);
    let duration_attrs: Vec<_> = points[0].attributes().cloned().collect();
    assert_eq!(
        attr_value(&duration_attrs, "gen_ai.provider.name").as_deref(),
        Some("stub-provider")
    );
    // No token-type attribute on the duration histogram.
    assert!(attr_value(&duration_attrs, "gen_ai.token.type").is_none());
    assert_eq!(
        points[0].bounds().collect::<Vec<_>>(),
        OPERATION_DURATION_BUCKET_BOUNDARIES.to_vec()
    );
}

#[test]
fn failed_chat_completion_records_neither_histogram() {
    let _guard = METRICS_TEST_MUTEX.lock().unwrap();
    let (provider, exporter) = harness();
    exporter.reset();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let client = ObservableChatClient::new(StubClient { fail: true }, "stub-provider");
        let _ = client
            .get_response(
                vec![Message::user("hi")],
                ChatOptions::new().with_model("request-model"),
            )
            .await;
    });

    provider.force_flush().unwrap();
    let finished = exporter.get_finished_metrics().unwrap();
    // Mirrors upstream: `_trace_get_response`'s exception branch never calls
    // `_capture_response`, so a failed call records no histogram data points
    // (the instruments may still appear if a prior test in this process
    // already recorded into them, but this run must not have added any).
    if let Some(token_usage) = find_metric(&finished, TOKEN_USAGE_METRIC) {
        let opentelemetry_sdk::metrics::data::AggregatedMetrics::U64(
            opentelemetry_sdk::metrics::data::MetricData::Histogram(hist),
        ) = token_usage.data()
        else {
            panic!("expected a u64 histogram");
        };
        assert_eq!(hist.data_points().count(), 0);
    }
    if let Some(duration) = find_metric(&finished, OPERATION_DURATION_METRIC) {
        let opentelemetry_sdk::metrics::data::AggregatedMetrics::F64(
            opentelemetry_sdk::metrics::data::MetricData::Histogram(hist),
        ) = duration.data()
        else {
            panic!("expected an f64 histogram");
        };
        assert_eq!(hist.data_points().count(), 0);
    }
}

#[test]
fn function_invocation_duration_records_tool_name_and_error_type() {
    let _guard = METRICS_TEST_MUTEX.lock().unwrap();
    let (provider, exporter) = harness();
    exporter.reset();

    metrics::record_function_invocation_duration("my_tool", Duration::from_millis(5), None);
    metrics::record_function_invocation_duration(
        "flaky_tool",
        Duration::from_millis(1),
        Some("tool"),
    );

    provider.force_flush().unwrap();
    let finished = exporter.get_finished_metrics().unwrap();

    let metric = find_metric(&finished, FUNCTION_INVOCATION_DURATION_METRIC)
        .unwrap_or_else(|| panic!("{FUNCTION_INVOCATION_DURATION_METRIC} not recorded"));
    assert_eq!(metric.unit(), "s");
    let opentelemetry_sdk::metrics::data::AggregatedMetrics::F64(
        opentelemetry_sdk::metrics::data::MetricData::Histogram(hist),
    ) = metric.data()
    else {
        panic!("expected an f64 histogram for {FUNCTION_INVOCATION_DURATION_METRIC}");
    };
    let points: Vec<_> = hist.data_points().collect();
    assert_eq!(points.len(), 2);

    let ok_point = points
        .iter()
        .find(|p| {
            p.attributes().any(|kv| {
                kv.key.as_str() == "agent_framework.function.name"
                    && kv.value.to_string() == "my_tool"
            })
        })
        .expect("my_tool data point");
    assert_eq!(ok_point.count(), 1);
    let ok_attrs: Vec<_> = ok_point.attributes().cloned().collect();
    assert!(attr_value(&ok_attrs, "error.type").is_none());

    let err_point = points
        .iter()
        .find(|p| {
            p.attributes().any(|kv| {
                kv.key.as_str() == "agent_framework.function.name"
                    && kv.value.to_string() == "flaky_tool"
            })
        })
        .expect("flaky_tool data point");
    let err_attrs: Vec<_> = err_point.attributes().cloned().collect();
    assert_eq!(
        attr_value(&err_attrs, "error.type").as_deref(),
        Some("tool")
    );
}
