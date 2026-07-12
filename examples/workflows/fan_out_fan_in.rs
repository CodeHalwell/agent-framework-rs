//! fan_out_fan_in: one source dispatches the same input to three independent
//! executors; their results fan back in to a single aggregator, which the
//! engine runs only once all three have delivered.
//!
//! Runs fully offline (no LLM calls) using `FunctionExecutor` nodes.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example fan_out_fan_in
//! ```

use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::FunctionExecutor;
use serde_json::{json, Value};

/// One fan-out branch: tags the shared input with this branch's note.
fn worker(id: &'static str, note: &'static str) -> Arc<dyn Executor> {
    Arc::new(FunctionExecutor::new(id, move |message, ctx| async move {
        let topic = message.as_str().unwrap_or_default();
        ctx.send_message(json!({ "executor": id, "note": format!("{note}: {topic}") }))
            .await?;
        Ok(())
    }))
}

#[tokio::main]
async fn main() -> Result<()> {
    let dispatch = FunctionExecutor::new("dispatch", |message, ctx| async move {
        ctx.send_message(message).await?;
        Ok(())
    });

    // The fan-in barrier: runs once, receiving all three branch results as a
    // single JSON array, in the declared source order.
    let aggregate = FunctionExecutor::new("aggregate", |message, ctx| async move {
        let Value::Array(items) = message else {
            return Err(Error::Workflow("aggregator expected an array".into()));
        };
        let mut report = String::from("Consolidated report:\n");
        for item in &items {
            report.push_str(&format!(
                "- {}\n",
                item["note"].as_str().unwrap_or_default()
            ));
        }
        ctx.yield_output(json!({ "report": report, "count": items.len() }))
            .await?;
        Ok(())
    });

    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(dispatch))
        .add_executor(worker("researcher", "market research"))
        .add_executor(worker("marketer", "positioning angle"))
        .add_executor(worker("legal", "compliance notes"))
        .add_executor(Arc::new(aggregate))
        .set_start("dispatch")
        .add_fan_out(
            "dispatch",
            vec![
                "researcher".to_string(),
                "marketer".to_string(),
                "legal".to_string(),
            ],
        )
        .add_fan_in(
            vec![
                "researcher".to_string(),
                "marketer".to_string(),
                "legal".to_string(),
            ],
            "aggregate",
        )
        .build()?;

    let run = workflow
        .run("budget electric bike for urban commuters")
        .await?;
    let output = run.last_output().expect("aggregate always yields output");
    println!("{}", output["report"].as_str().unwrap_or_default());
    assert_eq!(output["count"], json!(3));

    Ok(())
}
