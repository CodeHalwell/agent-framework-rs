//! Typed tools: `AiFunction::typed` derives a JSON Schema straight from a
//! `#[derive(Deserialize, JsonSchema)]` struct instead of a hand-written
//! `serde_json::Value` (compare with `agents/tools.rs`, which builds the
//! schema by hand).
//!
//! Runs offline: the schema-derivation and direct-invocation steps need no
//! network. Only the final step -- wiring the tool into a live `ChatAgent` --
//! is env-gated on `OPENAI_API_KEY` and skips gracefully without it.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example typed_tools
//! # with a live model too:
//! OPENAI_API_KEY=sk-... cargo run -p agent-framework-examples --example typed_tools
//! ```

use agent_framework::prelude::*;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

/// The tool's arguments. `Deserialize` lets `Tool::invoke` parse the model's
/// JSON arguments into this type; `JsonSchema` is what `AiFunction::typed`
/// uses to derive the parameters schema below -- no hand-written schema
/// needed.
#[derive(Debug, Deserialize, JsonSchema)]
struct WeatherArgs {
    /// The city to get the weather for.
    city: String,
    /// Preferred units: "celsius" or "fahrenheit". Defaults to "celsius".
    #[serde(default)]
    units: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // `AiFunction::typed` derives the parameters schema from `WeatherArgs`
    // via `schemars` at construction time, and deserializes the model's raw
    // JSON arguments into `WeatherArgs` before calling the closure.
    let get_weather = AiFunction::typed(
        "get_weather",
        "Get the current weather for a city.",
        |args: WeatherArgs| async move {
            let units = args.units.unwrap_or_else(|| "celsius".to_string());
            // A real implementation would call a weather API here.
            Ok(json!({ "city": args.city, "temperature": 21, "units": units }))
        },
    );

    // `parameters_schema()` (from the `Tool` trait) exposes exactly what
    // gets sent to the model: an object schema with `city` required and
    // `units` optional.
    println!("derived parameters schema:");
    println!(
        "{}\n",
        serde_json::to_string_pretty(&get_weather.parameters_schema())?
    );

    // Invoke the tool directly, exactly like the function-invocation loop
    // would: raw JSON in, raw JSON out. `Tool::invoke` takes `&self`, so this
    // doesn't consume `get_weather` -- it's still available below.
    let direct = get_weather
        .invoke(json!({ "city": "Lisbon", "units": "celsius" }))
        .await?;
    println!("direct invoke result: {direct}\n");

    // Wiring it into a live agent is the only part that needs a real model.
    let Ok(_) = std::env::var("OPENAI_API_KEY") else {
        println!("set OPENAI_API_KEY to also run this tool through a live ChatAgent");
        return Ok(());
    };

    let client = OpenAIClient::from_env("gpt-4o-mini")?;
    let agent = ChatAgent::builder(client)
        .name("weather-assistant")
        .instructions("You are a weather assistant. Use tools when needed.")
        .tool(get_weather.into_definition())
        .build();

    let response = agent.run_once("What's the weather in Lisbon?").await?;
    println!("agent: {}", response.text());

    Ok(())
}
