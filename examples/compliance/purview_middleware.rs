//! Microsoft Purview compliance middleware: `PurviewAgentMiddleware` sends
//! each prompt (and response) to Purview's Graph `processContent` API and
//! short-circuits the run with a configurable message when policy blocks it.
//! There is a `PurviewChatMiddleware` twin for the chat-client level.
//!
//! Skips gracefully unless configured:
//!   PURVIEW_TOKEN    a Graph API bearer token with ProtectionScopes/Content
//!                    permissions (acquisition is out of scope, as in Python)
//!   OPENAI_API_KEY   the model behind the agent
//!
//! ```bash
//! PURVIEW_TOKEN=eyJ... OPENAI_API_KEY=sk-... \
//! cargo run -p agent-framework-examples --example purview_middleware
//! ```

use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::purview::{PurviewSettings, StaticTokenProvider};

#[tokio::main]
async fn main() -> Result<()> {
    let (Ok(token), Ok(client)) = (
        std::env::var("PURVIEW_TOKEN"),
        OpenAIChatCompletionClient::from_env("gpt-4o-mini"),
    ) else {
        println!("set PURVIEW_TOKEN and OPENAI_API_KEY to run this example");
        return Ok(());
    };

    // `app_name` is the only required setting; tenant/location/version and
    // the blocked-prompt/response texts all have Python-parity defaults
    // (override via the with_* builders).
    let settings = PurviewSettings::new("agent-framework-rs-example");
    let middleware = PurviewAgentMiddleware::new(StaticTokenProvider::new(token), settings);

    // The middleware runs around the whole agent turn: the user prompt is
    // checked before the model is called, and the model's response before it
    // is returned. Blocked content never reaches the other side.
    let agent = Agent::builder(client)
        .name("compliant-assistant")
        .instructions("You are a helpful assistant.")
        .middleware(Arc::new(middleware))
        .build();

    let response = agent
        .run_once("Summarize our meeting notes policy.")
        .await?;
    println!("{}", response.text());

    Ok(())
}
