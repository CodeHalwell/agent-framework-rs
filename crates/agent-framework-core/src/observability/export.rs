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
//! Point it at an OpenTelemetry Collector, or — with the `azure-monitor`
//! feature — straight at Azure Monitor (Application Insights) with
//! [`OtelExport::with_azure_monitor`], the counterpart of upstream's
//! `configure_azure_monitor`. That destination uses the community
//! `opentelemetry-application-insights` exporter (there is no official
//! Microsoft OTel exporter for Rust) over the same blocking reqwest 0.13
//! client and TLS backend.
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
use opentelemetry::KeyValue;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::trace::SdkTracerProvider;
use opentelemetry_sdk::Resource;

/// Resource attribute marking telemetry as produced by the Agent Framework
/// (upstream #9589).
///
/// Azure's agent tooling — Foundry's telemetry views and the Application
/// Insights experiences built on them — keys off this attribute to tell
/// Agent Framework traces and metrics apart from an application's own. A
/// pipeline without it exports telemetry those views do not recognise.
///
/// The value is the string `"True"`, not a boolean, matching upstream's
/// `resource_attributes` default exactly: the two are distinct values on the
/// wire and a consumer matching the string would not match a bool.
const MICROSOFT_AGENT_FRAMEWORK_ATTR: &str = "microsoft.agent-framework";

/// The value for [`MICROSOFT_AGENT_FRAMEWORK_ATTR`]. See its note on why this
/// is the string `"True"` rather than a boolean.
const MICROSOFT_AGENT_FRAMEWORK_VALUE: &str = "True";

/// The OTel `Resource` shared by this pipeline's tracer and meter providers.
///
/// One resource for both signals, as upstream does: the marker attribute has
/// to be on traces and metrics alike, or Azure's views recognise only half of
/// what a run emits.
fn build_resource(service_name: &str) -> Resource {
    Resource::builder()
        .with_service_name(service_name.to_string())
        .with_attribute(KeyValue::new(
            MICROSOFT_AGENT_FRAMEWORK_ATTR,
            MICROSOFT_AGENT_FRAMEWORK_VALUE,
        ))
        .build()
}

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
    /// An Application Insights connection string: export there instead of
    /// to the OTLP endpoint.
    #[cfg(feature = "azure-monitor")]
    azure_monitor: Option<String>,
    traces: bool,
    metrics: bool,
    metric_interval: Duration,
    /// Extra span processors, run on every span ahead of the exporter.
    span_processors: Vec<SharedSpanProcessor>,
}

/// A span processor shared between clones of an [`OtelExport`].
#[derive(Debug, Clone)]
struct SharedSpanProcessor(std::sync::Arc<dyn opentelemetry_sdk::trace::SpanProcessor>);

impl opentelemetry_sdk::trace::SpanProcessor for SharedSpanProcessor {
    fn on_start(&self, span: &mut opentelemetry_sdk::trace::Span, cx: &opentelemetry::Context) {
        self.0.on_start(span, cx);
    }
    fn on_end(&self, span: opentelemetry_sdk::trace::SpanData) {
        self.0.on_end(span);
    }
    fn force_flush(&self) -> opentelemetry_sdk::error::OTelSdkResult {
        self.0.force_flush()
    }
    fn shutdown_with_timeout(&self, timeout: Duration) -> opentelemetry_sdk::error::OTelSdkResult {
        self.0.shutdown_with_timeout(timeout)
    }
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
            #[cfg(feature = "azure-monitor")]
            azure_monitor: None,
            traces: true,
            metrics: true,
            metric_interval: Duration::from_secs(60),
            span_processors: Vec::new(),
        }
    }

    /// Run `processor` on every span, ahead of the exporter (e.g. to enrich
    /// spans with deployment attributes). Processors run in the order added.
    pub fn with_span_processor(
        mut self,
        processor: impl opentelemetry_sdk::trace::SpanProcessor + 'static,
    ) -> Self {
        self.span_processors
            .push(SharedSpanProcessor(std::sync::Arc::new(processor)));
        self
    }

    /// Override the OTLP/HTTP endpoint (e.g. `http://collector:4318`).
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Export to Azure Monitor (Application Insights) with this connection
    /// string instead of to an OTLP endpoint. The string is checked when the
    /// pipeline is built.
    #[cfg(feature = "azure-monitor")]
    #[cfg_attr(docsrs, doc(cfg(feature = "azure-monitor")))]
    pub fn with_azure_monitor(mut self, connection_string: impl Into<String>) -> Self {
        self.azure_monitor = Some(connection_string.into());
        self
    }

    /// A pipeline exporting to Azure Monitor with the connection string in
    /// `APPLICATIONINSIGHTS_CONNECTION_STRING`, the variable upstream and the
    /// Azure Monitor distros read.
    #[cfg(feature = "azure-monitor")]
    #[cfg_attr(docsrs, doc(cfg(feature = "azure-monitor")))]
    pub fn azure_monitor_from_env(service_name: impl Into<String>) -> Result<Self> {
        let connection_string = std::env::var(APPLICATIONINSIGHTS_CONNECTION_STRING_ENV)
            .ok()
            .filter(|v| !v.trim().is_empty())
            .ok_or_else(|| {
                Error::Configuration(format!(
                    "{APPLICATIONINSIGHTS_CONNECTION_STRING_ENV} is not set."
                ))
            })?;
        Ok(Self::new(service_name).with_azure_monitor(connection_string))
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
        let resource = build_resource(&self.service_name);

        #[cfg(feature = "azure-monitor")]
        let azure = self
            .azure_monitor
            .as_deref()
            .map(azure_monitor_exporter)
            .transpose()?;

        let tracer_provider = if self.traces {
            let mut builder = SdkTracerProvider::builder().with_resource(resource.clone());
            for processor in &self.span_processors {
                builder = builder.with_span_processor(processor.clone());
            }
            #[cfg(feature = "azure-monitor")]
            let builder = match &azure {
                Some(exporter) => builder.with_batch_exporter(exporter.clone()),
                None => builder.with_batch_exporter(self.otlp_span_exporter()?),
            };
            #[cfg(not(feature = "azure-monitor"))]
            let builder = builder.with_batch_exporter(self.otlp_span_exporter()?);
            Some(builder.build())
        } else {
            None
        };

        let meter_provider = if self.metrics {
            let builder = SdkMeterProvider::builder().with_resource(resource);
            #[cfg(feature = "azure-monitor")]
            let builder = match &azure {
                Some(exporter) => builder.with_reader(
                    PeriodicReader::builder(exporter.clone())
                        .with_interval(self.metric_interval)
                        .build(),
                ),
                None => builder.with_reader(
                    PeriodicReader::builder(self.otlp_metric_exporter()?)
                        .with_interval(self.metric_interval)
                        .build(),
                ),
            };
            #[cfg(not(feature = "azure-monitor"))]
            let builder = builder.with_reader(
                PeriodicReader::builder(self.otlp_metric_exporter()?)
                    .with_interval(self.metric_interval)
                    .build(),
            );
            let provider = builder.build();
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

impl OtelExport {
    fn otlp_span_exporter(&self) -> Result<opentelemetry_otlp::SpanExporter> {
        opentelemetry_otlp::SpanExporter::builder()
            .with_http()
            .with_endpoint(signal_url(&self.endpoint, "/v1/traces"))
            .build()
            .map_err(|e| Error::other(format!("OTLP span exporter: {e}")))
    }

    fn otlp_metric_exporter(&self) -> Result<opentelemetry_otlp::MetricExporter> {
        opentelemetry_otlp::MetricExporter::builder()
            .with_http()
            .with_endpoint(signal_url(&self.endpoint, "/v1/metrics"))
            .build()
            .map_err(|e| Error::other(format!("OTLP metric exporter: {e}")))
    }
}

/// The environment variable holding an Application Insights connection
/// string.
#[cfg(feature = "azure-monitor")]
#[cfg_attr(docsrs, doc(cfg(feature = "azure-monitor")))]
pub const APPLICATIONINSIGHTS_CONNECTION_STRING_ENV: &str = "APPLICATIONINSIGHTS_CONNECTION_STRING";

/// The Application Insights exporter type this pipeline uses.
#[cfg(feature = "azure-monitor")]
type AzureMonitorExporter =
    opentelemetry_application_insights::Exporter<reqwest013::blocking::Client>;

#[cfg(feature = "azure-monitor")]
fn azure_monitor_exporter(connection_string: &str) -> Result<AzureMonitorExporter> {
    opentelemetry_application_insights::Exporter::new_from_connection_string(
        connection_string,
        reqwest013::blocking::Client::new(),
    )
    .map_err(|e| {
        Error::Configuration(format!(
            "invalid Application Insights connection string: {e}"
        ))
    })
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
    ///
    /// **Keep the [`OtelPipeline`] alive for as long as the subscriber is
    /// installed.** The layer holds a tracer belonging to this pipeline's
    /// tracer provider, and [`OtelExport::build`] installed its meter provider
    /// globally, so calling [`shutdown`](Self::shutdown) while the subscriber
    /// is still in place leaves it wired to stopped providers — every
    /// subsequent span and metric is then discarded, with nothing reporting
    /// it. Shut down on the way out of the process, not after installing.
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

    #[derive(Debug, Default)]
    struct Stamp(std::sync::Arc<std::sync::atomic::AtomicUsize>);

    impl opentelemetry_sdk::trace::SpanProcessor for Stamp {
        fn on_start(
            &self,
            span: &mut opentelemetry_sdk::trace::Span,
            _cx: &opentelemetry::Context,
        ) {
            use opentelemetry::trace::Span as _;
            span.set_attribute(KeyValue::new("stamped", true));
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        fn on_end(&self, _span: opentelemetry_sdk::trace::SpanData) {}
        fn force_flush(&self) -> opentelemetry_sdk::error::OTelSdkResult {
            Ok(())
        }
        fn shutdown_with_timeout(
            &self,
            _timeout: Duration,
        ) -> opentelemetry_sdk::error::OTelSdkResult {
            Ok(())
        }
    }

    #[tokio::test]
    async fn added_span_processors_see_every_span() {
        use opentelemetry::trace::{Tracer as _, TracerProvider as _};
        let count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let pipeline = OtelExport::new("svc")
            .with_endpoint("http://127.0.0.1:9")
            .with_metrics(false)
            .with_span_processor(Stamp(count.clone()))
            .build()
            .unwrap();
        let tracer = pipeline.tracer_provider.as_ref().unwrap().tracer("t");
        drop(tracer.start("one"));
        drop(tracer.start("two"));
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 2);
        let _ = pipeline.shutdown();
    }

    #[test]
    fn the_resource_marks_telemetry_as_agent_framework() {
        // Azure's Foundry and Application Insights agent views key off this
        // attribute to tell Agent Framework telemetry apart from the
        // application's own (upstream #9589). Without it a pipeline exports
        // telemetry those views do not recognise.
        let resource = build_resource("svc");
        let value = resource
            .get(&opentelemetry::Key::from_static_str(
                MICROSOFT_AGENT_FRAMEWORK_ATTR,
            ))
            .expect("the marker attribute must be on the resource");
        assert_eq!(value.as_str(), MICROSOFT_AGENT_FRAMEWORK_VALUE);
        // The string "True", not a boolean: a consumer matching the string
        // would not match a bool on the wire.
        assert_eq!(value.as_str(), "True");
    }

    #[test]
    fn the_resource_still_carries_the_service_name() {
        // The marker is additive — it must not displace the service name the
        // caller configured.
        let resource = build_resource("my-service");
        let value = resource
            .get(&opentelemetry::Key::from_static_str("service.name"))
            .expect("service.name must survive");
        assert_eq!(value.as_str(), "my-service");
    }

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

    #[cfg(feature = "azure-monitor")]
    #[test]
    fn a_malformed_connection_string_is_refused_at_build() {
        let err = OtelExport::new("svc")
            .with_azure_monitor("not a connection string")
            .with_metrics(false)
            .build()
            .expect_err("refused");
        assert!(err.to_string().contains("connection string"), "{err}");
    }

    #[cfg(feature = "azure-monitor")]
    #[test]
    fn a_valid_connection_string_builds_an_azure_monitor_pipeline() {
        let pipeline = OtelExport::new("svc")
            .with_azure_monitor(
                "InstrumentationKey=00000000-0000-0000-0000-000000000000;IngestionEndpoint=https://westeurope-5.in.applicationinsights.azure.com/",
            )
            .with_metrics(false)
            .build()
            .expect("builds");
        assert!(pipeline.tracer_provider.is_some());
    }

    /// Spans reach the Application Insights ingestion endpoint the connection
    /// string names, as `track` envelopes.
    #[cfg(feature = "azure-monitor")]
    #[test]
    fn spans_export_to_the_ingestion_endpoint() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(10)))
                .unwrap();
            let mut request = Vec::new();
            let mut buf = [0u8; 8192];
            // Read the head, then the body its Content-Length announces.
            loop {
                let n = stream.read(&mut buf).unwrap();
                request.extend_from_slice(&buf[..n]);
                let text = String::from_utf8_lossy(&request);
                if let Some(head_end) = text.find("\r\n\r\n") {
                    let length = text[..head_end]
                        .lines()
                        .find_map(|l| {
                            l.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .map(|v| v.trim().parse::<usize>().unwrap())
                        })
                        .unwrap_or(0);
                    if request.len() >= head_end + 4 + length || n == 0 {
                        break;
                    }
                }
            }
            let body = r#"{"itemsReceived":1,"itemsAccepted":1,"errors":[]}"#;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
            String::from_utf8_lossy(&request).into_owned()
        });

        // A connection string's endpoints are forced to https, so the
        // exporter built from one is pointed at this plain-HTTP server with
        // the (deprecated, but only) endpoint override.
        #[allow(deprecated)]
        let exporter = azure_monitor_exporter(
            "InstrumentationKey=00000000-0000-0000-0000-000000000000;IngestionEndpoint=https://example.invalid/",
        )
        .unwrap()
        .with_endpoint(&format!("http://127.0.0.1:{port}"))
        .unwrap();
        let provider = SdkTracerProvider::builder()
            .with_resource(build_resource("svc"))
            .with_batch_exporter(exporter)
            .build();
        {
            use opentelemetry::trace::{Span as _, Tracer as _};
            let mut span = provider.tracer(SCOPE_NAME).start("invoke_agent planner");
            span.end();
        }
        provider.shutdown().expect("flushed");
        let request = server.join().unwrap();
        assert!(request.starts_with("POST /v2"), "{request}");
        assert!(request.contains("/track"), "{request}");
    }
}
