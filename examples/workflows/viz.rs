//! Workflow visualization: render a branching graph as Mermaid and Graphviz
//! DOT text, ready to paste into any renderer that understands either format
//! (e.g. the Mermaid Live Editor, or `dot -Tpng`).
//!
//! Runs fully offline: building and inspecting a workflow's graph requires no
//! LLM calls, API key, or network access.
//!
//! ```bash
//! cargo run -p agent-framework --example workflow_viz
//! ```

use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::{Case, Default as SwitchDefault, FunctionExecutor};
use serde_json::json;

/// A no-op node, just to give the graph shape something to route through.
fn noop(id: &str) -> Arc<dyn Executor> {
    let id = id.to_string();
    Arc::new(FunctionExecutor::new(
        id,
        |_message, _ctx| async move { Ok(()) },
    ))
}

#[tokio::main]
async fn main() -> Result<()> {
    // Shape: "a" always notifies "b"; "a" also switches to "c" when the
    // message equals "hot", otherwise to "d" (the switch's default branch);
    // "c" and "d" both fan in to a "joiner" barrier.
    let workflow = WorkflowBuilder::new()
        .add_executor(noop("a"))
        .add_executor(noop("b"))
        .add_executor(noop("c"))
        .add_executor(noop("d"))
        .add_executor(noop("joiner"))
        .set_start("a")
        .add_conditional_edge("a", "b", |_msg| true)
        .add_switch(
            "a",
            vec![Case::labeled(|msg| msg == &json!("hot"), "c", "hot")],
            SwitchDefault::new("d"),
        )
        .add_fan_in(vec!["c".to_string(), "d".to_string()], "joiner")
        .name("branching-example")
        .build()?;

    println!("--- Mermaid (flowchart TD) ---\n");
    println!("{}", workflow.viz().to_mermaid());

    println!("\n--- Graphviz DOT ---\n");
    println!("{}", workflow.viz().to_dot());

    Ok(())
}
