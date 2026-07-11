//! Structured output: ask the model for a response conforming to a JSON
//! Schema, then parse it straight into a typed Rust struct.
//!
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run -p agent-framework-examples --example structured_output
//! ```

use agent_framework::prelude::*;
use serde::Deserialize;
use serde_json::json;

/// The shape we ask the model to produce. Only `Deserialize` is needed --
/// `parse_json` re-parses the response text into this type.
#[derive(Debug, Deserialize)]
struct Recipe {
    name: String,
    minutes: u32,
    ingredients: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIClient::from_env("gpt-4o-mini")?;

    // `ResponseFormat::json_schema` takes a raw JSON Schema value. Hand-writing
    // it is fine for a small shape like this; a larger app might derive one
    // with `schemars` instead.
    let schema = json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "minutes": { "type": "integer" },
            "ingredients": { "type": "array", "items": { "type": "string" } }
        },
        "required": ["name", "minutes", "ingredients"],
        "additionalProperties": false
    });

    let agent = ChatAgent::builder(client)
        .name("chef")
        .instructions("You suggest simple recipes.")
        .response_format(ResponseFormat::json_schema("Recipe", schema))
        .build();

    let response = agent
        .run_once("Suggest a quick weeknight pasta recipe")
        .await?;

    // `parse_json::<T>()` treats the response's concatenated text as JSON and
    // deserializes it into `T`. It mirrors Python's `response.value`.
    let recipe: Recipe = response.parse_json()?;
    println!(
        "{} ({} min)\n  ingredients: {}",
        recipe.name,
        recipe.minutes,
        recipe.ingredients.join(", ")
    );

    Ok(())
}
