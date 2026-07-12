//! MCP tools as a first-class `ToolSource`: `AgentBuilder::tool_source`
//! registers an `Arc<McpStdioTool>` directly, instead of snapshotting a
//! `Vec<ToolDefinition>` once at build time.
//!
//! Contrast with `mcp/mcp_tools.rs`, which calls `mcp.tool_definitions().await`
//! up front and hands the (now-frozen) result to `.tools(..)` -- the agent
//! never notices a later server-side tool-catalog change. Here, `.tool_source`
//! defers resolution to every `run`/`run_stream` call: the source connects
//! lazily on first use and thereafter serves a cached tool list that
//! self-invalidates when the server sends
//! `notifications/tools/list_changed`, so the agent always sees an up-to-date
//! catalog without ever being rebuilt.
//!
//! Prerequisite: a working `npx` (Node.js) on PATH, and network access the
//! first time (npm downloads and caches the server package). This spawns the
//! same `npx -y @modelcontextprotocol/server-everything` reference server as
//! `mcp_tools.rs`.
//!
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run -p agent-framework-examples --example mcp_first_class_tools
//! ```

use std::sync::Arc;

use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let Ok(client) = OpenAIClient::from_env("gpt-4o-mini") else {
        println!("set OPENAI_API_KEY to run this example");
        return Ok(());
    };

    let mcp = Arc::new(
        McpStdioTool::new("everything", "npx")
            .args(["-y", "@modelcontextprotocol/server-everything"])
            .description("Reference MCP server with demo tools (echo, add, ...)."),
    );

    // The manual approach (`mcp_tools.rs`) would be:
    //   let tools = mcp.tool_definitions().await?;
    //   Agent::builder(client).tools(tools).build()
    // -- a one-time snapshot taken before the agent exists. `.tool_source`
    // instead keeps the live `McpStdioTool` around and asks it, fresh, on
    // every run (see `ToolSource::resolve_tools`).
    let agent = Agent::builder(client)
        .name("assistant")
        .instructions("Use the available tools when they help answer the question.")
        .tool_source(mcp.clone())
        .build();

    // First run: `resolve_tools` connects lazily (handshake + `tools/list`)
    // and caches the result.
    let response = agent
        .run_once("Use the echo tool to repeat back: hello from a ToolSource")
        .await?;
    println!("{}", response.text());

    // Second run: the cache from the first run is reused -- no new
    // `tools/list` round trip unless the server signaled
    // `notifications/tools/list_changed` in between. Deliberately not naming
    // a specific tool here: the demo server's exact tool names/count can
    // change between releases (`npx -y` always fetches latest), and the
    // point is that the model picks whichever resolved tool fits, not that
    // this code hardcodes one.
    let response = agent
        .run_once("Now add 21 and 21 using whichever available tool fits.")
        .await?;
    println!("{}", response.text());

    mcp.close().await?;
    Ok(())
}
