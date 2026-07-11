//! sub_workflows: `WorkflowExecutor` embeds a whole child `Workflow` as a
//! single node in a parent graph. The child runs to completion inside that
//! node; its output is forwarded onward as an ordinary message, so the
//! parent graph can keep processing it just like any other executor's
//! output.
//!
//! Runs fully offline (no LLM calls) using `FunctionExecutor` nodes on both
//! sides.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example sub_workflows
//! ```

use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::FunctionExecutor;
use serde_json::json;

/// The child workflow: a two-step "word stats" pipeline (count -> format).
fn build_word_stats_workflow() -> Result<Workflow> {
    let count = FunctionExecutor::new("count", |message, ctx| async move {
        let text = message.as_str().unwrap_or_default().to_string();
        let words = text.split_whitespace().count();
        ctx.send_message(json!({ "text": text, "words": words }))
            .await?;
        Ok(())
    });
    let format = FunctionExecutor::new("format", |message, ctx| async move {
        let words = message["words"].as_i64().unwrap_or(0);
        let text = message["text"].as_str().unwrap_or_default();
        ctx.yield_output(json!(format!("\"{text}\" has {words} word(s)")))
            .await?;
        Ok(())
    });

    WorkflowBuilder::new()
        .add_executor(Arc::new(count))
        .add_executor(Arc::new(format))
        .set_start("count")
        .add_edge("count", "format")
        .name("word-stats")
        .build()
}

#[tokio::main]
async fn main() -> Result<()> {
    let child = build_word_stats_workflow()?;

    // "greet" announces the line before handing it to the embedded child
    // workflow; "report" receives whatever the child forwarded downstream.
    let greet = FunctionExecutor::new("greet", |message, ctx| async move {
        println!("parent: dispatching to sub-workflow: {message}");
        ctx.send_message(message).await?;
        Ok(())
    });
    let report = FunctionExecutor::new("report", |message, ctx| async move {
        ctx.yield_output(json!({ "from_child": message })).await?;
        Ok(())
    });

    let parent = WorkflowBuilder::new()
        .add_executor(Arc::new(greet))
        .add_executor(Arc::new(WorkflowExecutor::new("word_stats", child)))
        .add_executor(Arc::new(report))
        .set_start("greet")
        .add_chain(vec![
            "greet".to_string(),
            "word_stats".to_string(),
            "report".to_string(),
        ])
        .build()?;

    for line in [
        "Rust workflows compose.",
        "Sub-workflows nest cleanly inside a parent graph.",
    ] {
        let run = parent.run(line).await?;
        let output = run.last_output().expect("report always yields");
        println!("parent output: {}\n", output["from_child"]);
    }

    Ok(())
}
