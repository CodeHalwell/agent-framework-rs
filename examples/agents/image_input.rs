//! Image input: attach an image to a message two ways -- `Content::Uri` for
//! a remote URL, and `Content::Data` (built via `DataContent::from_bytes`)
//! for inline bytes, encoded as a `data:` URI.
//!
//! Env-gated on `OPENAI_API_KEY` (a vision-capable model, e.g. `gpt-4o-mini`,
//! is required to actually look at the image). Skips gracefully without it,
//! printing the message that would have been sent.
//!
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run -p agent-framework-examples --example image_input
//! ```

use agent_framework::prelude::*;
use agent_framework::types::{DataContent, UriContent};

/// The smallest possible valid PNG: a 1x1, 8-bit grayscale pixel (67 bytes).
/// Stands in for "a real screenshot/photo" so this example has no file-system
/// dependency.
const TINY_PNG: &[u8] = &[
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x3a, 0x7e, 0x9b,
    0x55, 0x00, 0x00, 0x00, 0x0a, 0x49, 0x44, 0x41, 0x54, 0x78, 0xda, 0x63, 0xf8, 0x0f, 0x00, 0x01,
    0x01, 0x01, 0x00, 0x1c, 0xb0, 0x8c, 0x99, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae,
    0x42, 0x60, 0x82,
];

#[tokio::main]
async fn main() -> Result<()> {
    // A remote image by URL...
    let remote = Content::Uri(UriContent {
        uri: "https://upload.wikimedia.org/wikipedia/commons/d/d3/Ferris.png".to_string(),
        media_type: "image/png".to_string(),
    });

    // ...and an inline image, base64-encoded into a `data:` URI from raw
    // bytes. `DataContent::from_bytes` does the encoding.
    let inline = Content::Data(DataContent::from_bytes(TINY_PNG, "image/png"));

    let message = Message::with_contents(
        Role::user(),
        vec![
            Content::text("What do you see in these two images?"),
            remote,
            inline,
        ],
    );

    let Ok(_) = std::env::var("OPENAI_API_KEY") else {
        println!("OPENAI_API_KEY not set -- skipping the live call. Here's the message that would be sent:");
        println!("{}", serde_json::to_string_pretty(&message)?);
        return Ok(());
    };

    let client = OpenAIClient::from_env("gpt-4o-mini")?;
    let agent = ChatAgent::builder(client)
        .name("vision-assistant")
        .instructions("Describe images concisely.")
        .build();

    let response = agent.run_once(vec![message]).await?;
    println!("{}", response.text());

    Ok(())
}
