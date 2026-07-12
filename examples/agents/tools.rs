//! SupportsAgentRun with a local tool. The framework runs the function-invocation loop
//! automatically: the model calls `get_weather`, the result is fed back, and
//! the model produces a final answer.
//!
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run -p agent-framework-examples --example tools
//! ```

use agent_framework::prelude::*;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIChatCompletionClient::from_env("gpt-4o-mini")?;

    let get_weather = FunctionTool::new(
        "get_weather",
        "Get the current weather for a city.",
        json!({
            "type": "object",
            "properties": { "city": { "type": "string", "description": "City name" } },
            "required": ["city"]
        }),
        |args| async move {
            let city = args["city"].as_str().unwrap_or("unknown");
            // A real implementation would call a weather API here.
            Ok(json!({ "city": city, "temperature_c": 21, "condition": "sunny" }))
        },
    )
    .into_definition();

    let agent = Agent::builder(client)
        .instructions("You are a weather assistant. Use tools when needed.")
        .tool(get_weather)
        .build();

    let response = agent.run_once("What's the weather in Paris?").await?;
    println!("{}", response.text());
    Ok(())
}
