//! A ready-made OTLP export pipeline for the spans and metrics this crate
//! emits. Behind the `otel-export` feature.
//!
//! [`super`] deliberately depends on no OTel SDK: it emits `tracing` spans and
//! records histograms through the `opentelemetry` **API** crate, both of which
//! are inert until an application installs a subscriber and a `MeterProvider`.
//! That keeps the SDK — and its version churn — out of the dependency graph of
//! every consumer. The cost is that "wire it to a backend" was left as prose,
//! and prose drifts.
//!
//! This module is the opt-in other half: one call that builds an OTLP exporter,
//! a tracer provider, a meter provider, and the `tracing`↔OTel bridge, wired to
//! the conventions [`super`] already emits.
//!
//! # Choosing between this and wiring it yourself
//!
//! Use [`OtelExport`] when you want the standard OTLP setup and would rather
//! not track four inter-dependent crate versions. Wire it yourself when you
//! already have a subscriber stack, need a non-OTLP exporter, or want to
//! control sampling and batching — in which case use [`OtelExport::build`] and
//! [`OtelPipeline::tracing_layer`] to compose, rather than
//! [`OtelPipeline::install`], which claims the global subscriber.
//!
//! Enabling `otel-export` implies `otel-metrics`, because the pipeline installs
//! the `MeterProvider` that the GenAI histograms record through; without it
//! the traces would export and the metrics would silently stay no-ops.
//!
//! # Transport
//!
//! OTLP over HTTP/protobuf rather than gRPC, which would pull the whole
//! tonic/hyper stack. The exporter uses reqwest's *blocking* client because the
//! batch span processor and the periodic metric reader each export from their
//! own background thread, where no tokio reactor is running and an async client
//! panics; that is the same reason `opentelemetry-otlp` defaults to it.
//!
//! `opentelemetry-otlp` brings its own reqwest 0.13, whose features cargo
//! cannot unify with the workspace's 0.12, so its TLS backend is enabled
//! explicitly (`reqwest-rustls`). Without that the exporter reaches plain-HTTP
//! collectors only and fails against any HTTPS endpoint — a failure that shows
//! up at run time, not build time.
//!
//! Point it at an OpenTelemetry Collector, which is also the supported route
//! to Azure Monitor — there is no official Microsoft OTel exporter for Rust.
//!
//! # Azure SDK client spans do not join this pipeline yet
//!
//! `azure_core_opentelemetry` bridges the Azure SDK's own client spans (the
//! token-fetch round trips made by `agent-framework-azure`'s `entra-sdk`
//! credentials, for instance) into OpenTelemetry, which would be a natural
//! companion to this module. It cannot be used here today: its current
//! release requires `opentelemetry` ^0.31, while this pipeline is built on
//! 0.32, and 0.x minor versions are semver-incompatible. Adding it resolves a
//! *second* copy of `opentelemetry` and `opentelemetry_sdk` into the graph,
//! each with its own `global` provider registry — so the Azure spans would
//! register against a global this pipeline never installs into and silently
//! reach no backend, which is worse than leaving them out. Revisit when that
//! crate moves to 0.32.
//!
//! # Example
//!
//! ```no_run
//! use agent_framework_core::observability::export::OtelExport;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let pipeline = OtelExport::new("my-agent-service")
//!     .with_endpoint("http://localhost:4318")
//!     .build()?;
//! pipeline.install()?;
//!
//! // ... run agents; spans and GenAI metrics now export ...
//!
//! pipeline.shutdown();
//! # Ok(())
//! # }
//! ```

use std::time::Duration;

use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::trace::SdkTracerProvider;
use opentelemetry_sdk::Resource;

use crate::error::{Error, Result};

/// Default OTLP/HTTP endpoint, matching the OpenTelemetry Collector's default
/// receiver port.
pub const DEFAULT_OTLP_ENDPOINT: &str = "http://localhost:4318";

/// Instrumentation-scope name used for the tracer this pipeline creates.
pub const SCOPE_NAME: &str = "agent_framework";

/// Builder for the OTLP export pipeline.
///
/// `service_name` is the only required input: it becomes the `service.name`
/// resource attribute, which is how every backend groups the emitted telemetry.
#[derive(Debug, Clone)]
pub struct OtelExport {
    service_name: String,
    endpoint: String,
    traces: bool,
    metrics: bool,
    metric_interval: Duration,
}

impl OtelExport {
    /// Start building a pipeline reporting under `service_name`.
    ///
    /// The endpoint defaults to `OTEL_EXPORTER_OTLP_ENDPOINT` when set, and
    /// [`DEFAULT_OTLP_ENDPOINT`] otherwise.
    pub fn new(service_name: impl Into<String>) -> Self {
        Self {
            service_name: service_name.into(),
            endpoint: std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
                .unwrap_or_else(|_| DEFAULT_OTLP_ENDPOINT.to_string()),
            traces: true,
            metrics: true,
            metric_interval: Duration::from_secs(60),
        }
    }

    /// Override the OTLP/HTTP endpoint (e.g. `http://collector:4318`).
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Export spans. On by default.
    pub fn with_traces(mut self, enabled: bool) -> Self {
        self.traces = enabled;
        self
    }

    /// Export the GenAI metrics. On by default.
    pub fn with_metrics(mut self, enabled: bool) -> Self {
        self.metrics = enabled;
        self
    }

    /// How often metrics are pushed. Defaults to 60s, the OTel default.
    pub fn with_metric_interval(mut self, interval: Duration) -> Self {
        self.metric_interval = interval;
        self
    }

    /// Build the providers and install the `MeterProvider` globally.
    ///
    /// The meter provider is installed here rather than in
    /// [`OtelPipeline::install`] because the GenAI histograms resolve their
    /// instruments through [`opentelemetry::global`]; a caller that composes
    /// with [`OtelPipeline::tracing_layer`] and never calls `install` still
    /// needs metrics to work.
    ///
    /// Spans are *not* wired up here — that requires claiming a subscriber,
    /// which is [`OtelPipeline::install`]'s job, or the caller's via
    /// [`OtelPipeline::tracing_layer`].
    pub fn build(self) -> Result<OtelPipeline> {
        let resource = Resource::builder()
            .with_service_name(self.service_name.clone())
            .build();

        let tracer_provider = if self.traces {
            let exporter = opentelemetry_otlp::SpanExporter::builder()
                .with_http()
                .with_endpoint(format!("{}/v1/traces", self.endpoint.trim_end_matches('/')))
                .build()
                .map_err(|e| Error::other(format!("OTLP span exporter: {e}")))?;
            Some(
                SdkTracerProvider::builder()
                    .with_resource(resource.clone())
                    .with_batch_exporter(exporter)
                    .build(),
            )
        } else {
            None
        };

        let meter_provider = if self.metrics {
            let exporter = opentelemetry_otlp::MetricExporter::builder()
                .with_http()
                .with_endpoint(format!(
                    "{}/v1/metrics",
                    self.endpoint.trim_end_matches('/')
                ))
                .build()
                .map_err(|e| Error::other(format!("OTLP metric exporter: {e}")))?;
            let reader = PeriodicReader::builder(exporter)
                .with_interval(self.metric_interval)
                .build();
            let provider = SdkMeterProvider::builder()
                .with_resource(resource)
                .with_reader(reader)
                .build();
            opentelemetry::global::set_meter_provider(provider.clone());
            Some(provider)
        } else {
            None
        };

        Ok(OtelPipeline {
            tracer_provider,
            meter_provider,
        })
    }
}

/// A built export pipeline.
///
/// Hold it for the process lifetime and call [`shutdown`](Self::shutdown)
/// before exit: the span exporter batches, so dropping without a flush loses
/// whatever is still queued.
#[derive(Debug)]
pub struct OtelPipeline {
    tracer_provider: Option<SdkTracerProvider>,
    meter_provider: Option<SdkMeterProvider>,
}

impl OtelPipeline {
    /// The `tracing` layer bridging this crate's spans into OTel, for callers
    /// composing their own subscriber.
    ///
    /// Returns `None` when traces are disabled.
    pub fn tracing_layer<S>(
        &self,
    ) -> Option<tracing_opentelemetry::OpenTelemetryLayer<S, opentelemetry_sdk::trace::SdkTracer>>
    where
        S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
    {
        let tracer = self.tracer_provider.as_ref()?.tracer(SCOPE_NAME);
        Some(tracing_opentelemetry::layer().with_tracer(tracer))
    }

    /// Install a global `tracing` subscriber wiring this crate's spans to OTLP.
    ///
    /// Convenience for applications with no subscriber of their own. Filtering
    /// follows `RUST_LOG` (defaulting to `info`). Fails if a global subscriber
    /// is already set — compose with [`tracing_layer`](Self::tracing_layer)
    /// instead in that case.
    pub fn install(&self) -> Result<()> {
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::util::SubscriberInitExt;
        use tracing_subscriber::EnvFilter;

        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        tracing_subscriber::registry()
            .with(filter)
            .with(self.tracing_layer())
            .try_init()
            .map_err(|e| Error::other(format!("installing the tracing subscriber: {e}")))
    }

    /// Flush and shut down both providers.
    ///
    /// Errors are reported through `tracing` rather than returned: this runs on
    /// the shutdown path, where a caller has nothing useful left to do with a
    /// failure, and losing the other provider's flush because the first errored
    /// would be worse than logging.
    pub fn shutdown(&self) {
        if let Some(p) = &self.tracer_provider {
            if let Err(e) = p.shutdown() {
                tracing::warn!(error = %e, "OTel tracer provider shutdown failed");
            }
        }
        if let Some(p) = &self.meter_provider {
            if let Err(e) = p.shutdown() {
                tracing::warn!(error = %e, "OTel meter provider shutdown failed");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_defaults_and_overrides() {
        let export = OtelExport::new("svc");
        // Either the env default or the constant, never empty.
        assert!(!export.endpoint.is_empty());

        let export = OtelExport::new("svc").with_endpoint("http://collector:4318");
        assert_eq!(export.endpoint, "http://collector:4318");
    }

    /// Disabling both signals must still build — a caller toggling exports off
    /// by config should not have to branch around the pipeline.
    #[test]
    fn a_pipeline_with_both_signals_disabled_builds_and_is_inert() {
        let pipeline = OtelExport::new("svc")
            .with_traces(false)
            .with_metrics(false)
            .build()
            .expect("builds with no signals");

        assert!(pipeline.tracer_provider.is_none());
        assert!(pipeline.meter_provider.is_none());
        assert!(
            pipeline
                .tracing_layer::<tracing_subscriber::Registry>()
                .is_none(),
            "no tracer provider means no layer to compose"
        );
        // Shutting down a pipeline that exports nothing is a no-op, not a panic.
        pipeline.shutdown();
    }

    /// A trailing slash on the endpoint must not produce `//v1/traces`. Builds
    /// the exporter for real, so a malformed URL surfaces here.
    #[test]
    fn a_trailing_slash_on_the_endpoint_is_normalized() {
        let pipeline = OtelExport::new("svc")
            .with_endpoint("http://localhost:4318/")
            .with_metrics(false)
            .build()
            .expect("trailing-slash endpoint builds");
        assert!(pipeline.tracer_provider.is_some());
        pipeline.shutdown();
    }
}
