//! MCP prompts: list a server's reusable prompt templates (`prompts/list`),
//! render one into messages (`prompts/get`), and feed the result straight
//! into a `ChatAgent` run as ordinary `Message`s.
//!
//! Same prerequisites as `mcp_tools.rs`: a working `npx` (Node.js) on PATH,
//! plus OPENAI_API_KEY to run the rendered prompt through a real model. This
//! connects to the same `@modelcontextprotocol/server-everything` reference
//! server, which exposes a couple of demo prompts. If a server declares no
//! prompts at all (or doesn't advertise the `prompts` capability),
//! `.prompts()` short-circuits to an empty list without a round trip -- this
//! example handles that case gracefully too, rather than assuming any
//! particular prompt exists.
//!
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run -p agent-framework-examples --example mcp_prompts
//! ```

use agent_framework::prelude::*;
use serde_json::{json, Map, Value};

#[tokio::main]
async fn main() -> Result<()> {
    let Ok(client) = OpenAIClient::from_env("gpt-4o-mini") else {
        println!("set OPENAI_API_KEY to run this example");
        return Ok(());
    };

    let mcp = McpStdioTool::new("everything", "npx")
        .args(["-y", "@modelcontextprotocol/server-everything"])
        .description("Reference MCP server with demo prompts.");

    // Connects lazily (if needed) and lists the server's prompts, cached
    // until a `notifications/prompts/list_changed` invalidates it.
    let prompts = mcp.prompts().await?;
    if prompts.is_empty() {
        // Either the server didn't declare the `prompts` capability during
        // `initialize`, or it declared it with none registered -- `.prompts()`
        // returns `[]` without ever issuing a `prompts/list` request in that
        // case (same short-circuit as `.load_prompts(false)`).
        println!("server exposes no prompts (or doesn't support them) -- nothing to render.");
        mcp.close().await?;
        return Ok(());
    }

    println!("discovered {} prompt(s):", prompts.len());
    for p in &prompts {
        println!(
            "  - {}: {}",
            p.name,
            p.description.as_deref().unwrap_or("(no description)")
        );
    }

    // Render the first prompt. A real caller would collect actual values for
    // any `required` arguments; the demo server tolerates empty strings.
    let first = &prompts[0];
    let mut arguments = Map::new();
    for arg in first.arguments.iter().flatten() {
        arguments.insert(arg.name.clone(), json!(""));
    }
    let messages = mcp
        .get_prompt(&first.name, Value::Object(arguments))
        .await?;
    println!(
        "\nprompt '{}' rendered {} message(s):",
        first.name,
        messages.len()
    );
    for m in &messages {
        println!("  [{}] {}", m.role, m.text());
    }

    // The rendered messages are ordinary `Message`s -- hand them straight
    // to an agent run, exactly like a hand-written prompt would be.
    let agent = ChatAgent::builder(client)
        .name("assistant")
        .instructions("Respond helpfully to the rendered prompt.")
        .build();
    let response = agent.run(messages, None).await?;
    println!("\nagent: {}", response.text());

    mcp.close().await?;
    Ok(())
}
