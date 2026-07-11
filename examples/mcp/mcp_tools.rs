//! Model Context Protocol (MCP) tools: connect to an MCP server over stdio,
//! list its tools, and wire them straight into a `ChatAgent`.
//!
//! Prerequisite: a working `npx` (Node.js) on PATH, and network access the
//! first time (npm downloads and caches the server package). This spawns
//! `npx -y @modelcontextprotocol/server-everything`, a reference MCP server
//! published by the Model Context Protocol project with a handful of
//! harmless demo tools (echo, add, ...).
//!
//! Requires the `mcp` feature:
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run -p agent-framework-examples --example mcp_tools
//! ```

use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let mcp = McpStdioTool::new("everything", "npx")
        .args(["-y", "@modelcontextprotocol/server-everything"])
        .description("Reference MCP server with demo tools (echo, add, ...).");

    // Spawns the server subprocess (if not already connected), performs the
    // MCP `initialize` handshake, and lists its tools as `ToolDefinition`s --
    // ready to hand to a `ChatAgent` exactly like a local `AiFunction`.
    let tools = mcp.tool_definitions().await?;
    println!("discovered {} MCP tool(s):", tools.len());
    for tool in &tools {
        println!("  - {}: {}", tool.name, tool.description);
    }

    let client = OpenAIClient::from_env("gpt-4o-mini")?;
    let agent = ChatAgent::builder(client)
        .name("assistant")
        .instructions("You can use the available tools when they help answer the question.")
        .tools(tools)
        .build();

    let response = agent
        .run_once("Use the echo tool to repeat back: hello from MCP")
        .await?;
    println!("\n{}", response.text());

    mcp.close().await?;
    Ok(())
}
