//! Hosted tools: capabilities the *provider* runs on its own servers, not
//! your process. You declare them; the service executes them and folds the
//! results into the response.
//!
//! The framework ships one constructor per kind:
//!
//! | Constructor | `ToolKind` | Runs |
//! | --- | --- | --- |
//! | `hosted_web_search()` | `HostedWebSearch` | provider-side web search |
//! | `hosted_code_interpreter()` | `HostedCodeInterpreter` | provider-side sandbox |
//! | `hosted_file_search(max_results)` | `HostedFileSearch` | provider-side vector stores |
//! | `hosted_image_generation()` | `HostedImageGeneration` | provider-side image model |
//! | `hosted_mcp(name, url, allowed)` | `HostedMcp` | an MCP server the *provider* connects to |
//!
//! The contrast worth internalising: a `FunctionTool` is executed by the
//! function-invocation loop inside your process, so you see every call and
//! can gate it with `ApprovalMode`. A hosted tool never reaches your process
//! at all -- which is why `hosted_mcp` carries its own `McpApprovalMode`, the
//! *service's* approval gate, and why the results come back as dedicated
//! `Content` variants (`SearchToolCall`, `CodeInterpreterToolResult`,
//! `McpServerToolCall`, ...) rather than `FunctionResult`.
//!
//! Not every provider supports every kind; the converters drop what their
//! API cannot express, so declaring one is safe but not a guarantee. Runs
//! offline (it inspects the definitions rather than calling a provider); set
//! `OPENAI_API_KEY` to also fire a real hosted web search.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example hosted_tools
//! OPENAI_API_KEY=sk-... cargo run -p agent-framework-examples --example hosted_tools
//! ```

use agent_framework::prelude::*;

/// A one-line summary of what a definition will put on the wire.
fn describe(tool: &ToolDefinition) -> String {
    match &tool.kind {
        ToolKind::Function => {
            "Function            -- executed locally by the invocation loop".into()
        }
        ToolKind::HostedCodeInterpreter => "HostedCodeInterpreter -- provider-side sandbox".into(),
        ToolKind::HostedImageGeneration => {
            "HostedImageGeneration -- provider-side image model".into()
        }
        ToolKind::HostedWebSearch => "HostedWebSearch     -- provider-side web search".into(),
        ToolKind::HostedFileSearch { max_results } => {
            format!(
                "HostedFileSearch    -- provider-side vector stores, max_results={max_results:?}"
            )
        }
        ToolKind::HostedMcp { url, allowed_tools } => format!(
            "HostedMcp           -- the *service* connects to {url}, allowed={allowed_tools:?}"
        ),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("== the five hosted-tool constructors ==\n");

    let tools = vec![
        hosted_web_search(),
        hosted_code_interpreter(),
        // Cap how many chunks the service's retrieval returns; `None` leaves
        // it to the provider's default.
        hosted_file_search(Some(5)),
        hosted_image_generation(),
        // A hosted MCP connector: the provider dials the MCP server itself,
        // so your process never speaks MCP. Restrict it to named tools rather
        // than handing the model the server's whole surface.
        hosted_mcp(
            "docs",
            "https://mcp.example.com/sse",
            Some(vec!["search_docs".to_string()]),
        ),
    ];

    for tool in &tools {
        println!("  {:<20} {}", tool.name, describe(tool));
    }

    println!("\n== approval: local vs. hosted ==\n");

    // A local tool's approval gate is enforced by *this* process: the run
    // pauses and hands you a `FunctionApprovalRequestContent` (see the
    // `approvals` example).
    let local = FunctionTool::new(
        "delete_record",
        "Delete a record by id.",
        serde_json::json!({
            "type": "object",
            "properties": { "id": { "type": "string" } },
            "required": ["id"]
        }),
        |args| async move { Ok(serde_json::json!({ "deleted": args["id"] })) },
    )
    .into_definition()
    .with_approval_mode(ApprovalMode::AlwaysRequire);
    println!(
        "  local  `{}`: approval_mode={:?} -- enforced here, the run pauses for you",
        local.name, local.approval_mode
    );

    // A hosted MCP connector's gate is enforced by the *service*, so it is a
    // separate setting that travels on the wire in the tool's parameters
    // rather than being interpreted locally.
    let hosted = hosted_mcp("docs", "https://mcp.example.com/sse", None)
        .mcp_approval_mode(McpApprovalMode::Always);
    println!(
        "  hosted `{}`: wire approval_mode={} -- enforced by the provider",
        hosted.name, hosted.parameters["approval_mode"]
    );

    // `PerTool` splits the difference: gate the destructive calls, let the
    // read-only ones through without a round trip to a human.
    let per_tool = hosted_mcp("repo", "https://mcp.example.com/repo", None).mcp_approval_mode(
        McpApprovalMode::PerTool {
            always: vec!["delete_branch".to_string()],
            never: vec!["list_branches".to_string()],
        },
    );
    println!(
        "  hosted `{}`: wire approval_mode={}",
        per_tool.name, per_tool.parameters["approval_mode"]
    );

    println!("\n== attaching them to an agent ==\n");
    println!(
        "  Agent::builder(client).tools([hosted_web_search(), hosted_code_interpreter()])\n  \
         -- mixed freely with local FunctionTools in the same list."
    );

    // Provider support, as implemented by this workspace's converters (an
    // unsupported kind is dropped with a `tracing::warn!` rather than
    // failing the request). See `providers/anthropic_hosted_tools.rs` and
    // `providers/azure_foundry_bing_grounding.rs` for live runs.
    println!("\n== provider support, per this workspace's converters ==\n");
    for (provider, supported) in [
        (
            "OpenAI (Responses API)",
            "web search, code interpreter, file search, image generation, MCP",
        ),
        (
            "Azure OpenAI (Responses)",
            "as OpenAI, subject to what the deployment enables",
        ),
        (
            "Azure AI Foundry",
            "all five (carried through on the Responses wire format)",
        ),
        (
            "Anthropic",
            "web search, code execution, MCP (as `mcp_servers`); file search dropped",
        ),
        (
            "Gemini",
            "web search (`googleSearch`), code interpreter (`codeExecution`)",
        ),
        (
            "Chat Completions / others",
            "none -- Chat Completions has no wire form for hosted tools",
        ),
    ] {
        println!("  {provider:<26} {supported}");
    }

    live_web_search().await
}

/// Optionally run a real hosted web search, when a key is present.
async fn live_web_search() -> Result<()> {
    let Ok(key) = std::env::var("OPENAI_API_KEY") else {
        println!(
            "\n(set OPENAI_API_KEY to also run a real hosted web search through the \
             Responses API)"
        );
        return Ok(());
    };
    let _ = key;

    println!("\n== live: hosted web search via the OpenAI Responses API ==\n");
    // Hosted tools need the Responses API client: Chat Completions has no
    // wire representation for them.
    let client = OpenAIChatClient::from_env("gpt-4o-mini")?;
    let agent = Agent::builder(client)
        .name("researcher")
        .instructions("Answer using the web search tool. Cite what you find.")
        .tool(hosted_web_search())
        .build();

    let response = agent
        .run_once("What did the Rust project release most recently?")
        .await?;
    println!("{}", response.text());

    // Hosted results arrive as their own content variants, not FunctionResult.
    let hosted_contents: Vec<&str> = response
        .messages
        .iter()
        .flat_map(|m| m.contents.iter())
        .filter_map(|c| match c {
            Content::SearchToolCall(_) => Some("SearchToolCall"),
            Content::SearchToolResult(_) => Some("SearchToolResult"),
            Content::CodeInterpreterToolCall(_) => Some("CodeInterpreterToolCall"),
            Content::CodeInterpreterToolResult(_) => Some("CodeInterpreterToolResult"),
            Content::McpServerToolCall(_) => Some("McpServerToolCall"),
            Content::McpServerToolResult(_) => Some("McpServerToolResult"),
            _ => None,
        })
        .collect();
    println!("\nhosted-tool content items in the response: {hosted_contents:?}");

    Ok(())
}
