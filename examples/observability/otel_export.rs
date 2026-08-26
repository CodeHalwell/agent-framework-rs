//! Exporting this crate's spans and GenAI metrics to a real backend, via the
//! `otel-export` feature.
//!
//! `agent-framework-core` emits `tracing` spans following the OpenTelemetry
//! GenAI semantic conventions, and records its GenAI histograms through the
//! `opentelemetry` **API** crate. Both are inert until something exports them.
//! This example shows the two ways to close that loop:
//!
//! * **Route A — the built-in pipeline.** `OtelExport` builds an OTLP
//!   exporter, a tracer provider, a meter provider and the `tracing`↔OTel
//!   bridge, then claims the global subscriber. One call, nothing to version.
//! * **Route B — compose it yourself.** Take the same pipeline's
//!   `tracing_layer()` and add it to a subscriber stack you own, alongside a
//!   fmt layer, your own filters, or other layers.
//!
//! This file is compiled by CI, so the wiring here cannot drift out of date
//! the way a prose snippet can.
//!
//! It runs fully offline. With no collector listening the exporter cannot
//! deliver, and the final flush reports that — which is exactly what happens
//! in production when a collector is down, and is why `shutdown` returns its
//! outcome instead of logging into a subscriber that is already being torn
//! down. Point it somewhere real with `OTEL_EXPORTER_OTLP_ENDPOINT`, or run
//! one locally:
//!
//! ```bash
//! docker run --rm -p 4318:4318 otel/opentelemetry-collector
//! cargo run -p agent-framework-examples --example otel_export
//! ```
//!
//! There is no official Microsoft OpenTelemetry exporter for Rust, so the
//! supported route to Azure Monitor is OTLP into an OpenTelemetry Collector
//! configured with the Azure Monitor exporter — which is what this points at.

use agent_framework::observability::export::OtelExport;
use agent_framework::observability::ObservableChatClient;
use agent_framework::prelude::*;
use async_trait::async_trait;

/// A minimal offline stand-in for a model, so the example needs no API key.
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

/// Route B, kept as a compiled function rather than a comment.
///
/// `tracing_layer()` hands back the OTel bridge layer so it can sit in a
/// subscriber you assemble — here beside a `fmt` layer, so spans go to a
/// collector *and* stay readable on the console. Use this instead of
/// `install()` whenever the application already owns its subscriber, since
/// `install()` sets the global default and would conflict.
#[allow(dead_code)]
fn compose_your_own_subscriber() -> Result<()> {
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let pipeline = OtelExport::new("my-agent-service").build()?;

    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new("info"))
        .with(tracing_subscriber::fmt::layer())
        .with(pipeline.tracing_layer())
        .try_init()
        .map_err(|e| agent_framework_core::error::Error::other(e.to_string()))?;

    pipeline.shutdown()?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Route A: build the pipeline and let it install the subscriber. The
    // endpoint defaults to OTEL_EXPORTER_OTLP_ENDPOINT, else localhost:4318.
    //
    // Order matters, which is why this is the first thing main does: the GenAI
    // instruments are created once on first use and bind to whichever meter
    // provider is installed at that moment. Run a chat call before build() and
    // they bind to the no-op provider for the life of the process — traces
    // would still export while metrics stayed silently empty.
    let pipeline = OtelExport::new("agent-framework-example")
        .with_metric_interval(std::time::Duration::from_secs(5))
        .build()?;
    pipeline.install()?;

    // Everything below is ordinary framework usage — no telemetry-specific
    // code. `ObservableChatClient` emits the `chat` span; the agent run emits
    // `invoke_agent`; both carry the GenAI attributes, and the histograms
    // record through the MeterProvider the pipeline installed.
    let client = ObservableChatClient::new(CannedClient, "demo-provider");
    let agent = Agent::builder(client)
        .instructions("You are concise.")
        .build();

    for question in ["What is the answer?", "Are you sure?"] {
        let reply = agent.run_once(question).await?;
        println!("{question} -> {}", reply.text());
    }

    // Flush before exit: the span exporter batches, so dropping without this
    // loses whatever is still queued.
    //
    // `shutdown` returns the outcome instead of logging it, because by this
    // point the subscriber installed above can no longer report anything —
    // its only layer feeds the tracer provider being shut down. Handle it
    // rather than propagate: with no collector listening this *does* fail, and
    // failing to deliver telemetry is not a reason to exit non-zero. What
    // matters is that the caller can now see it at all.
    if let Err(e) = pipeline.shutdown() {
        eprintln!("telemetry flush failed (expected with no collector running): {e}");
    }
    Ok(())
}
