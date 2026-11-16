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
    let names = Names::default();
    let subscriber = tracing_subscriber::registry().with(names.clone());
    let _guard = tracing::subscriber::set_default(subscriber);
    let workflow = WorkflowBuilder::new()
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

/// The run span follows the OTel GenAI conventions: `invoke_workflow {name}`
/// for a named workflow, `gen_ai.workflow.name` for the name itself, and
/// `gen_ai.operation.name` on both (upstream #9900, .NET #9899).
#[tokio::test(flavor = "current_thread")]
async fn the_run_span_is_named_after_the_workflow_and_the_genai_operation() {
    /// Collects the run span's fields, both those set when it opens and those
    /// filled in later with `Span::record`.
    #[derive(Clone, Default)]
    struct Fields {
        seen: Arc<Mutex<Vec<(String, String)>>>,
        run_spans: Arc<Mutex<Vec<tracing::span::Id>>>,
    }

    struct Probe<'a>(&'a mut Vec<(String, String)>);
    impl tracing::field::Visit for Probe<'_> {
        fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
            self.0.push((field.name().to_string(), value.to_string()));
        }
        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            self.0
                .push((field.name().to_string(), format!("{value:?}")));
        }
    }

    impl<S: tracing::Subscriber> Layer<S> for Fields {
        fn on_new_span(
            &self,
            attrs: &tracing::span::Attributes<'_>,
            id: &tracing::span::Id,
            _ctx: Context<'_, S>,
        ) {
            if attrs.metadata().name() != "invoke_workflow" {
                return;
            }
            self.run_spans.lock().unwrap().push(id.clone());
            let mut out = Vec::new();
            attrs.record(&mut Probe(&mut out));
            self.seen.lock().unwrap().extend(out);
        }

        fn on_record(
            &self,
            id: &tracing::span::Id,
            values: &tracing::span::Record<'_>,
            _ctx: Context<'_, S>,
        ) {
            if !self.run_spans.lock().unwrap().contains(id) {
                return;
            }
            let mut out = Vec::new();
            values.record(&mut Probe(&mut out));
            self.seen.lock().unwrap().extend(out);
        }
    }

    async fn fields_of(name: Option<&str>) -> Vec<(String, String)> {
        let fields = Fields::default();
        let names = Names::default();
        let subscriber = tracing_subscriber::registry()
            .with(fields.clone())
            .with(names.clone());
        let _guard = tracing::subscriber::set_default(subscriber);
        let mut builder = WorkflowBuilder::new()
            .add_executor(Arc::new(FunctionExecutor::new(
                "a",
                |msg, ctx| async move { ctx.yield_output(msg).await },
            )))
            .set_start("a");
        if let Some(name) = name {
            builder = builder.name(name);
        }
        let workflow = builder.build().unwrap();
        workflow.run(serde_json::json!("hi")).await.unwrap();
        let mut out = fields.seen.lock().unwrap().clone();
        // The span-name probe sees the final `otel.name` too; keep both so an
        // unnamed run still yields the bare name.
        for n in names.0.lock().unwrap().iter() {
            if n.starts_with("invoke_workflow") {
                out.push(("otel.name".to_string(), n.clone()));
            }
        }
        out
    }

    let named = fields_of(Some("triage")).await;
    let get = |fields: &[(String, String)], key: &str| {
        fields
            .iter()
            .filter(|(k, v)| k == key && !v.is_empty())
            .map(|(_, v)| v.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(get(&named, "gen_ai.operation.name"), ["invoke_workflow"]);
    assert_eq!(get(&named, "gen_ai.workflow.name"), ["triage"]);
    assert!(
        get(&named, "otel.name").contains(&"invoke_workflow triage".to_string()),
        "{named:?}"
    );
    assert!(
        get(&named, "workflow.name").is_empty(),
        "the pre-#9900 attribute must be gone: {named:?}"
    );

    // An unnamed workflow reports no name attribute and a bare span name.
    let unnamed = fields_of(None).await;
    assert_eq!(get(&unnamed, "gen_ai.operation.name"), ["invoke_workflow"]);
    assert!(
        get(&unnamed, "gen_ai.workflow.name").is_empty(),
        "an unnamed workflow must not report an empty name: {unnamed:?}"
    );
    assert!(
        get(&unnamed, "otel.name").contains(&"invoke_workflow".to_string()),
        "{unnamed:?}"
    );
}
