//! switch_case: `add_switch` evaluates its cases in order and routes to the
//! first match, falling back to the default branch -- unlike
//! `add_conditional_edge`, exactly one downstream executor runs.
//!
//! Runs fully offline (no LLM calls) using `FunctionExecutor` nodes.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example switch_case
//! ```

use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::{Case, Default as SwitchDefault, FunctionExecutor};
use serde_json::json;

/// A priority bucket that reports the score it received.
fn bucket(name: &'static str) -> Arc<dyn Executor> {
    Arc::new(FunctionExecutor::new(
        name,
        move |message, ctx| async move {
            ctx.yield_output(json!({ "bucket": name, "score": message }))
                .await?;
            Ok(())
        },
    ))
}

#[tokio::main]
async fn main() -> Result<()> {
    let classify = FunctionExecutor::new("classify", |message, ctx| async move {
        ctx.send_message(message).await?;
        Ok(())
    });

    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(classify))
        .add_executor(bucket("high"))
        .add_executor(bucket("medium"))
        .add_executor(bucket("low"))
        .set_start("classify")
        .add_switch(
            "classify",
            vec![
                Case::labeled(|msg| msg.as_i64().unwrap_or(0) >= 80, "high", "score >= 80"),
                Case::labeled(
                    |msg| msg.as_i64().unwrap_or(0) >= 40,
                    "medium",
                    "score >= 40",
                ),
            ],
            SwitchDefault::new("low"),
        )
        .build()?;

    for score in [95, 55, 10] {
        let run = workflow.run(json!(score)).await?;
        let output = run.last_output().expect("every score lands in a bucket");
        println!("score {score:>3} -> {}", output["bucket"].as_str().unwrap());
    }

    Ok(())
}
