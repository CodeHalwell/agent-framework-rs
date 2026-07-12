//! Anthropic hosted tools: `hosted_web_search()` gives Claude a server-side
//! web-search tool (no local execution or result plumbing needed -- the
//! service performs the search itself). `.max_uses(n)` caps how many
//! searches a single request may perform; it's read by Anthropic only (the
//! OpenAI/Azure AI Foundry converters ignore it).
//!
//! Anthropic also offers a hosted code-execution tool
//! (`hosted_code_interpreter()`, mapped to `code_execution_20250825`) --
//! wire it in exactly the same way, via `.tool(hosted_code_interpreter())`;
//! this example only calls out its `ToolDefinition` without spending an API
//! call on it.
//!
//! Skips gracefully unless `ANTHROPIC_API_KEY` is set.
//!
//! ```bash
//! ANTHROPIC_API_KEY=sk-ant-... cargo run -p agent-framework-examples --example anthropic_hosted_tools
//! ```

use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let Ok(_) = std::env::var("ANTHROPIC_API_KEY") else {
        println!("set ANTHROPIC_API_KEY to run this example");
        return Ok(());
    };

    let client = AnthropicClient::from_env("claude-sonnet-4-5-20250929")?;

    let web_search = hosted_web_search().max_uses(3);

    let agent = Agent::builder(client)
        .name("research-assistant")
        .instructions("Use web search for anything time-sensitive; cite your sources briefly.")
        .tool(web_search)
        .build();

    let response = agent
        .run_once("What is the latest stable version of the Rust compiler?")
        .await?;
    println!("{}\n", response.text());

    // Print any citation annotations the search tool attached, if present.
    for m in &response.messages {
        for content in &m.contents {
            if let Content::Text(t) = content {
                for a in t.annotations.iter().flatten() {
                    if let Some(url) = &a.url {
                        println!("  source: {} ({})", url, a.title.as_deref().unwrap_or(""));
                    }
                }
            }
        }
    }

    // Also available (not called here, to avoid spending a code-execution
    // request): Anthropic's hosted code-execution tool.
    let code_exec = hosted_code_interpreter();
    println!("\n(also available via the same pattern: {code_exec:?})");

    Ok(())
}
