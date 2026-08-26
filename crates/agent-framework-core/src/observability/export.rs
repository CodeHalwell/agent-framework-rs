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
//! // Flush before exit; a failure here means queued telemetry was dropped.
//! pipeline.shutdown()?;
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

/// Join a configured endpoint with a signal path.
///
/// `OTEL_EXPORTER_OTLP_ENDPOINT` is a *base* URL under the OTel specification,
/// with the signal path appended — which is what this does. But the
/// signal-specific variables (`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` and friends)
/// are full URLs, and a caller who reaches for [`OtelExport::with_endpoint`]
/// having read those docs will naturally pass one. Appending unconditionally
/// would turn that into `…/v1/traces/v1/traces`, which fails as an opaque
/// exporter error far from the mistake. An endpoint already carrying its
/// signal path is therefore used as given.
///
/// Note this port does not read the signal-specific variables itself; only
/// `OTEL_EXPORTER_OTLP_ENDPOINT` is consulted, in [`OtelExport::new`].
///
/// One pipeline drives both signals from one endpoint, so a signal path found
/// on the input is stripped before the right one is appended, rather than
/// merely suppressed. Suppressing would leave a traces URL correct for traces
/// and nonsense for metrics (`…/v1/traces/v1/metrics`) — the same class of
/// broken URL, just harder to notice because only one signal breaks.
///
/// Only the path is touched. A query or fragment is detached first and put
/// back afterwards, because appending to the whole string would bury the
/// signal path inside the query — `https://gw/otlp?api-version=1` would become
/// `https://gw/otlp?api-version=1/v1/traces`, which is a valid URL pointing at
/// the wrong path with a corrupted parameter, so it fails as a remote 404
/// rather than anything that names the cause. Query strings carrying an auth
/// token or an api-version are common enough on hosted collectors to be worth
/// handling. Split by hand rather than pulling in a URL parser: the endpoint is
/// only ever reassembled here, never inspected, so parsing would buy nothing
/// but a dependency.
fn signal_url(endpoint: &str, signal_path: &str) -> String {
    const SIGNAL_PATHS: [&str; 3] = ["/v1/traces", "/v1/metrics", "/v1/logs"];

    // Fragment first: it may itself contain a `?`, which is fragment text.
    let (rest, fragment) = match endpoint.split_once('#') {
        Some((rest, fragment)) => (rest, Some(fragment)),
        None => (endpoint, None),
    };
    let (path, query) = match rest.split_once('?') {
        Some((path, query)) => (path, Some(query)),
        None => (rest, None),
    };

    let mut base = path.trim_end_matches('/');
    for known in SIGNAL_PATHS {
        if let Some(stripped) = base.strip_suffix(known) {
            base = stripped.trim_end_matches('/');
            break;
        }
    }

    let mut out = format!("{base}{signal_path}");
    if let Some(query) = query {
        out.push('?');
        out.push_str(query);
    }
    if let Some(fragment) = fragment {
        out.push('#');
        out.push_str(fragment);
    }
    out
}

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
    ///
    /// Turning this off also makes [`OtelPipeline::install`] a no-op, so a
    /// metrics-only pipeline leaves the global `tracing` subscriber free for
    /// the application's own logging.
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
    /// **Call this before running any instrumented code.** The GenAI
    /// instruments are created once, on first use, and bind to whichever
    /// meter provider is installed at that moment — so a chat call made
    /// before `build()` permanently binds them to the no-op provider, and
    /// metrics stay silently inert for the life of the process while traces
    /// export normally. Nothing reports this; it looks like the metrics were
    /// never recorded.
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
                .with_endpoint(signal_url(&self.endpoint, "/v1/traces"))
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
                .with_endpoint(signal_url(&self.endpoint, "/v1/metrics"))
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
    ///
    /// With traces disabled ([`OtelExport::with_traces(false)`](OtelExport::with_traces))
    /// this does nothing and returns `Ok`, deliberately. There is no span layer
    /// to install, so the subscriber it would otherwise register is a bare
    /// registry that records nothing — and, worse, claiming the global slot is
    /// irreversible, so a metrics-only pipeline would silently take away the
    /// application's ability to install its own logging subscriber. Returning
    /// an error instead was the alternative, but a caller wiring this from
    /// configuration (`with_traces(cfg.traces)`) would then have to branch
    /// around a case where nothing is wrong. Metrics are unaffected either way:
    /// their provider is installed by [`OtelExport::build`], not here.
    pub fn install(&self) -> Result<()> {
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::util::SubscriberInitExt;
        use tracing_subscriber::EnvFilter;

        // Nothing to wire, and claiming the global subscriber anyway would cost
        // the application its own.
        if self.tracer_provider.is_none() {
            return Ok(());
        }

        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        tracing_subscriber::registry()
            .with(filter)
            .with(self.tracing_layer())
            .try_init()
            .map_err(|e| Error::other(format!("installing the tracing subscriber: {e}")))
    }

    /// Flush and shut down both providers.
    ///
    /// Both are always attempted, and their errors aggregated: a failure
    /// flushing one must not cost the other its flush, so this does not
    /// short-circuit.
    ///
    /// The outcome is *returned* rather than logged, because logging it would
    /// go nowhere in the very setup this module recommends.
    /// [`install`](Self::install) builds a subscriber of an `EnvFilter` and the
    /// OpenTelemetry layer and nothing else — there is no formatting layer for
    /// a `tracing` event to reach, and the one layer present forwards to a
    /// tracer provider this method has just shut down. A failed flush means
    /// queued telemetry was dropped, which is worth telling the caller about;
    /// `Result` is `#[must_use]`, so ignoring it has to be deliberate.
    pub fn shutdown(&self) -> Result<()> {
        let mut failures = Vec::new();

        if let Some(p) = &self.tracer_provider {
            if let Err(e) = p.shutdown() {
                failures.push(format!("tracer provider: {e}"));
            }
        }
        if let Some(p) = &self.meter_provider {
            if let Err(e) = p.shutdown() {
                failures.push(format!("meter provider: {e}"));
            }
        }

        if failures.is_empty() {
            Ok(())
        } else {
            Err(Error::other(format!(
                "OTel shutdown failed, so queued telemetry may have been dropped: {}",
                failures.join("; ")
            )))
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
        // Shutting down a pipeline that exports nothing is a no-op, and
        // reports success rather than a spurious failure.
        assert!(pipeline.shutdown().is_ok());
    }

    #[test]
    fn signal_paths_are_joined_without_doubling() {
        // The documented base-URL form.
        assert_eq!(
            signal_url("http://localhost:4318", "/v1/traces"),
            "http://localhost:4318/v1/traces"
        );
        // A trailing slash must not produce `//v1/traces`.
        assert_eq!(
            signal_url("http://localhost:4318/", "/v1/metrics"),
            "http://localhost:4318/v1/metrics"
        );
        // A full signal URL, as the signal-specific OTel env vars carry, is
        // used as given rather than suffixed a second time.
        assert_eq!(
            signal_url("http://collector:4318/v1/traces", "/v1/traces"),
            "http://collector:4318/v1/traces"
        );
        // ...and the *other* signal is rebased off it rather than stacked on
        // top, since one pipeline serves both from a single endpoint.
        assert_eq!(
            signal_url("http://collector:4318/v1/traces", "/v1/metrics"),
            "http://collector:4318/v1/metrics"
        );
        // A path-prefixed collector keeps its prefix.
        assert_eq!(
            signal_url("http://gateway/otlp", "/v1/traces"),
            "http://gateway/otlp/v1/traces"
        );
    }

    /// A query string must survive on the *query* side of the URL. Appending to
    /// the whole string instead buries the signal path in the query, producing
    /// a valid URL that targets the wrong path with a corrupted parameter —
    /// which surfaces only as a remote 404.
    #[test]
    fn a_query_or_fragment_survives_the_signal_path() {
        assert_eq!(
            signal_url("https://gateway/otlp?api-version=1", "/v1/traces"),
            "https://gateway/otlp/v1/traces?api-version=1"
        );
        // Rebasing across signals has to work with a query attached too.
        assert_eq!(
            signal_url("https://collector/v1/traces?token=abc", "/v1/metrics"),
            "https://collector/v1/metrics?token=abc"
        );
        // A fragment is preserved, and a `?` inside it stays fragment text
        // rather than being mistaken for the start of a query.
        assert_eq!(
            signal_url("https://gateway/otlp#frag?not-a-query", "/v1/traces"),
            "https://gateway/otlp/v1/traces#frag?not-a-query"
        );
        // Both at once, in the correct order.
        assert_eq!(
            signal_url("https://gateway/otlp?a=1#f", "/v1/metrics"),
            "https://gateway/otlp/v1/metrics?a=1#f"
        );
    }

    /// A metrics-only pipeline must leave the global subscriber alone, so the
    /// application can still install its own logging.
    ///
    /// Asserted by installing twice: claiming the global slot is irreversible,
    /// so if the first call had taken it the second would fail with "a global
    /// default trace dispatcher has already been set". Both succeeding is only
    /// possible if neither claimed it. (This also holds if some other test in
    /// this binary claimed the global first — without the guard both calls
    /// would fail instead.)
    #[test]
    fn a_metrics_only_pipeline_does_not_claim_the_global_subscriber() {
        let pipeline = OtelExport::new("svc")
            .with_traces(false)
            .build()
            .expect("metrics-only pipeline builds");

        pipeline.install().expect("first install is a no-op");
        pipeline
            .install()
            .expect("so is the second, having taken nothing");

        let _ = pipeline.shutdown();
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
        let _ = pipeline.shutdown();
    }
}
