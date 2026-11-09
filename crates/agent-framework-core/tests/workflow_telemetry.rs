//! Workflow spans: build, run, executor processing, message sends and edge
//! groups, each of which `WorkflowTelemetryOptions` can turn off.

use std::sync::{Arc, Mutex};

use agent_framework_core::workflow::{FunctionExecutor, WorkflowBuilder, WorkflowTelemetryOptions};
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::Layer;

#[derive(Clone, Default)]
struct Names(Arc<Mutex<Vec<String>>>);

impl<S: tracing::Subscriber> Layer<S> for Names {
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        _id: &tracing::span::Id,
        _ctx: Context<'_, S>,
    ) {
        let mut otel_name = None;
        struct Probe<'a>(&'a mut Option<String>);
        impl tracing::field::Visit for Probe<'_> {
            fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                if field.name() == "otel.name" {
                    *self.0 = Some(value.to_string());
                }
            }
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                if field.name() == "otel.name" {
                    *self.0 = Some(format!("{value:?}"));
                }
            }
        }
        attrs.record(&mut Probe(&mut otel_name));
        if let Some(name) = otel_name {
            self.0.lock().unwrap().push(name);
        }
    }
}

async fn run(options: WorkflowTelemetryOptions) -> Vec<String> {
    run_named(options, None).await
}

async fn run_named(options: WorkflowTelemetryOptions, name: Option<&str>) -> Vec<String> {
    let names = Names::default();
    let subscriber = tracing_subscriber::registry().with(names.clone());
    let _guard = tracing::subscriber::set_default(subscriber);
    let mut builder = WorkflowBuilder::new();
    if let Some(name) = name {
        builder = builder.name(name);
    }
    let workflow = builder
        .add_executor(Arc::new(FunctionExecutor::new(
            "a",
            |msg, ctx| async move { ctx.send_message(msg).await },
        )))
        .add_executor(Arc::new(FunctionExecutor::new(
            "b",
            |msg, ctx| async move { ctx.yield_output(msg).await },
        )))
        .set_start("a")
        .add_edge("a", "b")
        .with_telemetry(options)
        .build()
        .unwrap();
    workflow.run(serde_json::json!("hi")).await.unwrap();
    let names = names.0.lock().unwrap().clone();
    names
}

#[tokio::test(flavor = "current_thread")]
async fn a_run_emits_the_workflow_spans() {
    let names = run(WorkflowTelemetryOptions::default()).await;
    for expected in [
        "workflow.build",
        "invoke_workflow",
        "executor.process a",
        "executor.process b",
        "message.send",
        "edge_group.process SingleEdgeGroup",
    ] {
        assert!(
            names.iter().any(|n| n == expected),
            "{expected} missing from {names:?}"
        );
    }
}

#[tokio::test(flavor = "current_thread")]
async fn each_span_kind_can_be_turned_off() {
    let names = run(WorkflowTelemetryOptions {
        disable_workflow_build: true,
        disable_workflow_run: true,
        disable_executor_process: true,
        disable_edge_group_process: true,
        disable_message_send: true,
        enable_sensitive_data: false,
    })
    .await;
    assert!(names.is_empty(), "{names:?}");
}

/// A named workflow suffixes its run span, per the OTel GenAI conventions
/// upstream aligned on in #9900 / #9899 (`invoke_workflow {name}`).
#[tokio::test(flavor = "current_thread")]
async fn a_named_workflow_suffixes_the_run_span() {
    let names = run_named(WorkflowTelemetryOptions::default(), Some("triage")).await;
    assert!(
        names.iter().any(|n| n == "invoke_workflow triage"),
        "{names:?}"
    );
    // The bare name is the unnamed form; a named run must not also emit it.
    assert!(!names.iter().any(|n| n == "invoke_workflow"), "{names:?}");
}
