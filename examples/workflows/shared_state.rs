//! shared_state: pass a lightweight reference through the message graph
//! while the actual payload -- and a running audit trail -- live in
//! `SharedState`, visible to every executor in the run.
//!
//! Runs fully offline (no LLM calls) using `FunctionExecutor` nodes.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example shared_state
//! ```

use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::FunctionExecutor;
use serde_json::{json, Value};

const DOC: &str = "Rust workflows make shared state easy to reason about";

/// Append `step` to the run-scoped "audit_trail" list kept in `SharedState`.
async fn record_visit(ctx: &WorkflowContext, step: &str) {
    let step = step.to_string();
    ctx.shared_state()
        .update("audit_trail", move |current| {
            let mut trail: Vec<Value> = current
                .and_then(|v| v.as_array().cloned())
                .unwrap_or_default();
            trail.push(json!(step));
            json!(trail)
        })
        .await;
}

#[tokio::main]
async fn main() -> Result<()> {
    // Stores the (potentially large) document once and forwards only its id
    // downstream, instead of copying the text through every message.
    let ingest = FunctionExecutor::new("ingest", |message, ctx| async move {
        let doc_id = "doc-1".to_string();
        ctx.shared_state()
            .set(format!("doc:{doc_id}"), message)
            .await;
        record_visit(&ctx, "ingest").await;
        ctx.send_message(json!(doc_id)).await?;
        Ok(())
    });

    let word_count = FunctionExecutor::new("word_count", |message, ctx| async move {
        let doc_id = message.as_str().unwrap_or_default().to_string();
        let text = ctx
            .shared_state()
            .get(&format!("doc:{doc_id}"))
            .await
            .and_then(|v| v.as_str().map(str::to_string))
            .unwrap_or_default();
        let words = text.split_whitespace().count();
        ctx.shared_state()
            .set(format!("stats:{doc_id}"), json!({ "words": words }))
            .await;
        record_visit(&ctx, "word_count").await;
        ctx.send_message(json!(doc_id)).await?;
        Ok(())
    });

    let finalize = FunctionExecutor::new("finalize", |message, ctx| async move {
        let doc_id = message.as_str().unwrap_or_default().to_string();
        let shared = ctx.shared_state();
        let doc = shared
            .get(&format!("doc:{doc_id}"))
            .await
            .unwrap_or_default();
        let stats = shared
            .get(&format!("stats:{doc_id}"))
            .await
            .unwrap_or_default();
        record_visit(&ctx, "finalize").await;
        let trail = shared.get("audit_trail").await.unwrap_or_default();
        ctx.yield_output(json!({ "doc": doc, "stats": stats, "audit_trail": trail }))
            .await?;
        Ok(())
    });

    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(ingest))
        .add_executor(Arc::new(word_count))
        .add_executor(Arc::new(finalize))
        .set_start("ingest")
        .add_chain(vec![
            "ingest".to_string(),
            "word_count".to_string(),
            "finalize".to_string(),
        ])
        .build()?;

    let run = workflow.run(DOC).await?;
    let output = run.last_output().unwrap();
    println!("doc: {}", output["doc"].as_str().unwrap());
    println!("stats: {}", output["stats"]);
    println!("audit trail: {}", output["audit_trail"]);
    assert_eq!(
        output["stats"]["words"],
        json!(DOC.split_whitespace().count())
    );

    Ok(())
}
