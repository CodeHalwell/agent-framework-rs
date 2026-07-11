//! conditional_edges: a single triage executor routes each message to one or
//! more downstream executors via `add_conditional_edge`, keyed off the
//! message content. Unlike `add_switch`, every edge whose condition holds
//! fires -- there is no "first match wins" exclusivity.
//!
//! Runs fully offline (no LLM calls) using `FunctionExecutor` nodes.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example conditional_edges
//! ```

use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::FunctionExecutor;
use serde_json::{json, Value};

/// A downstream team that reports which ticket it picked up.
fn team(name: &'static str) -> Arc<dyn Executor> {
    Arc::new(FunctionExecutor::new(
        name,
        move |message, ctx| async move {
            ctx.yield_output(json!({ "handled_by": name, "ticket": message }))
                .await?;
            Ok(())
        },
    ))
}

/// A condition matching tickets whose text contains `word`.
fn mentions(word: &'static str) -> impl Fn(&Value) -> bool + Send + Sync + 'static {
    move |msg| {
        msg.as_str()
            .map(|s| s.to_lowercase().contains(word))
            .unwrap_or(false)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let triage = FunctionExecutor::new("triage", |message, ctx| async move {
        ctx.send_message(message).await?;
        Ok(())
    });

    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(triage))
        .add_executor(team("billing_team"))
        .add_executor(team("tech_team"))
        .add_executor(team("general_team"))
        .set_start("triage")
        .add_conditional_edge("triage", "billing_team", mentions("invoice"))
        .add_conditional_edge("triage", "tech_team", mentions("error"))
        .add_conditional_edge("triage", "general_team", |msg: &Value| {
            let text = msg.as_str().unwrap_or_default().to_lowercase();
            !text.contains("invoice") && !text.contains("error")
        })
        .build()?;

    for ticket in [
        "My invoice total looks wrong this month.",
        "The app throws an error when I try to log in.",
        "What are your support hours?",
    ] {
        let run = workflow.run(ticket).await?;
        let outputs = run.outputs();
        assert_eq!(
            outputs.len(),
            1,
            "each sample ticket matches exactly one team"
        );
        let output = &outputs[0];
        println!(
            "{:>12} -> {}",
            output["handled_by"].as_str().unwrap(),
            output["ticket"].as_str().unwrap()
        );
    }

    Ok(())
}
