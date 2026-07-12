//! MCP roots: advertise a static list of filesystem roots to a server via
//! `.roots(...)`.
//!
//! Configuring any roots turns on the `roots` capability during
//! `initialize`. From then on, if the server ever sends a `roots/list`
//! request -- typically to discover which directories it should treat as the
//! caller's workspace -- this crate answers it automatically from the list
//! configured here; there is nothing further to wire up. (This crate only
//! supports a *static* root list: there's no `notifications/roots/list_changed`
//! support, so `listChanged` is always advertised as `false`.)
//!
//! Prerequisite: a working `npx` (Node.js) on PATH. Root exposure works with
//! any server; this example connects to the same
//! `@modelcontextprotocol/server-everything` reference server the other MCP
//! examples use so it can also show the discovered tools, and (env-gated
//! like the others) run one through a real model.
//!
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run -p agent-framework-examples --example mcp_roots
//! ```

use agent_framework::mcp::Root;
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let Ok(client) = OpenAIChatCompletionClient::from_env("gpt-4o-mini") else {
        println!("set OPENAI_API_KEY to run this example");
        return Ok(());
    };

    let mcp = McpStdioTool::new("everything", "npx")
        .args(["-y", "@modelcontextprotocol/server-everything"])
        .roots([
            Root::new("file:///workspace").with_name("Workspace"),
            Root::new("file:///tmp").with_name("Scratch"),
        ]);

    // `connect()` performs the `initialize` handshake. Because roots were
    // configured above, that handshake advertises `capabilities.roots`; any
    // `roots/list` request the server sends afterward is answered from the
    // two roots below without any callback on our side.
    mcp.connect().await?;
    println!("connected -- advertised roots capability with:");
    println!("  - file:///workspace (\"Workspace\")");
    println!("  - file:///tmp (\"Scratch\")");
    println!(
        "(the roots/list flow is server-initiated: a server that scopes file access to a \n\
         workspace -- e.g. a filesystem server -- calls it on its own; this reference server \n\
         doesn't happen to, but the capability above is answered automatically the moment it does.)"
    );

    let tools = mcp.tool_definitions().await?;
    println!("\n{} tool(s) available:", tools.len());
    for tool in &tools {
        println!("  - {}", tool.name);
    }

    let agent = Agent::builder(client)
        .name("assistant")
        .instructions("Use the available tools when they help answer the question.")
        .tools(tools)
        .build();
    let response = agent
        .run_once("Use the echo tool to repeat back: roots configured")
        .await?;
    println!("\n{}", response.text());

    mcp.close().await?;
    Ok(())
}
