//! Graph validation: `WorkflowBuilder::build` refuses to hand back a
//! workflow whose graph could not run correctly, and `validate_workflow_graph`
//! exposes the same checks with a typed `ValidationType` you can branch on.
//!
//! `build()` folds a failure into `Error::Workflow` with the validation
//! error's `Display` as the message, which is the right shape for an
//! application that just wants to surface the problem. A tool that builds
//! graphs from user input (a UI, a declarative loader, a linter) usually
//! wants the *category* instead, so it can point at the offending edge or
//! offer a fix — that is what calling the validator directly gives you.
//!
//! Each failure below is deliberately constructed, then caught and printed.
//! Runs fully offline.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example workflow_validation
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::{
    validate_workflow_graph, EdgeGroup, FunctionExecutor, ValidationType,
};
use serde_json::json;

/// A do-nothing node: enough to be a registered executor in the graph.
fn node(id: &'static str) -> Arc<dyn Executor> {
    Arc::new(FunctionExecutor::new(id, move |message, ctx| async move {
        ctx.send_message(message).await?;
        Ok(())
    }))
}

/// A terminal node that yields its input as the workflow output.
fn sink(id: &'static str) -> Arc<dyn Executor> {
    Arc::new(FunctionExecutor::new(id, move |message, ctx| async move {
        ctx.yield_output(message).await?;
        Ok(())
    }))
}

/// Try to build `builder` and report what happened.
fn try_build(label: &str, builder: WorkflowBuilder) {
    match builder.build() {
        Ok(_) => println!("  {label:<24} built OK"),
        Err(e) => println!("  {label:<24} rejected: {e}"),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("== the five validation categories, as `build()` reports them ==\n");

    // 1. StartNotRegistered -- `set_start` names a node that was never added.
    try_build(
        "start not registered",
        WorkflowBuilder::new()
            .add_executor(node("a"))
            .set_start("does_not_exist"),
    );

    // 2. UnknownExecutor -- an edge points at a node that is not in the graph.
    try_build(
        "edge to unknown node",
        WorkflowBuilder::new()
            .add_executor(node("a"))
            .set_start("a")
            .add_edge("a", "ghost"),
    );

    // 3. EdgeDuplication -- the same directed edge declared twice. Usually a
    //    copy-paste slip; left in place it would deliver the message twice.
    try_build(
        "duplicate edge",
        WorkflowBuilder::new()
            .add_executor(node("a"))
            .add_executor(sink("b"))
            .set_start("a")
            .add_edge("a", "b")
            .add_edge("a", "b"),
    );

    // 4. GraphConnectivity -- a node exists but nothing can ever reach it, so
    //    it would silently never run.
    try_build(
        "unreachable node",
        WorkflowBuilder::new()
            .add_executor(node("a"))
            .add_executor(sink("b"))
            .add_executor(sink("orphan"))
            .set_start("a")
            .add_edge("a", "b"),
    );

    // 5. OutputValidation -- a node named as both a final and an intermediate
    //    output, which cannot be both.
    try_build(
        "output/intermediate clash",
        WorkflowBuilder::new()
            .add_executor(node("a"))
            .add_executor(sink("b"))
            .set_start("a")
            .add_edge("a", "b")
            .output_from(["b"])
            .intermediate_output_from(["b"]),
    );

    // And the graph that passes all five.
    try_build(
        "a valid graph",
        WorkflowBuilder::new()
            .add_executor(node("a"))
            .add_executor(sink("b"))
            .set_start("a")
            .add_edge("a", "b"),
    );

    println!("\n== the same checks, with the category in hand ==\n");

    // `validate_workflow_graph` takes the raw graph pieces, so a tool that
    // assembles a graph from somewhere other than the builder (a YAML spec,
    // a visual editor) can validate before it ever constructs one.
    let mut executors: HashMap<String, Arc<dyn Executor>> = HashMap::new();
    executors.insert("a".into(), node("a"));
    executors.insert("b".into(), sink("b"));
    executors.insert("orphan".into(), sink("orphan"));

    let edge_groups = vec![EdgeGroup::Single {
        source: "a".into(),
        target: "b".into(),
        condition: None,
    }];

    match validate_workflow_graph(&executors, &edge_groups, "a", &[], &[]) {
        Ok(()) => println!("  graph is valid"),
        Err(e) => {
            // The typed category is what lets a caller react rather than just
            // report: highlight the node, suggest the missing edge, and so on.
            let advice = match e.validation_type {
                ValidationType::StartNotRegistered => "add the start node, or fix set_start",
                ValidationType::UnknownExecutor => "add_executor for the node the edge names",
                ValidationType::EdgeDuplication => "delete the repeated add_edge call",
                ValidationType::GraphConnectivity => "connect it, or delete it",
                ValidationType::OutputValidation => "pick one of output_from / intermediate_output_from",
            };
            println!("  category: {:?}", e.validation_type);
            println!("  wire name: {}", e.validation_type);
            println!("  message:  {}", e.message);
            println!("  fix:      {advice}");
        }
    }

    println!("\n== validation runs before anything executes ==\n");

    // A valid graph, run for real, to make the contrast concrete: `build()`
    // having succeeded is the guarantee that the shape is sound. It says
    // nothing about what the nodes *do* -- a node that panics or returns an
    // error still fails at run time.
    let workflow = WorkflowBuilder::new()
        .name("validated")
        .add_executor(node("a"))
        .add_executor(sink("b"))
        .set_start("a")
        .add_edge("a", "b")
        .build()?;

    let run = workflow.run(json!("hello")).await?;
    println!("  ran the valid graph -> output {:?}", run.last_output());
    println!("  graph signature: {}", workflow.graph_signature());
    println!(
        "\n  (that signature is what checkpoint resume validates against -- see\n  \
         the `workflow_checkpoint` example.)"
    );

    Ok(())
}
