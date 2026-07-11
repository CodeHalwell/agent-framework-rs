//! `AzureOpenAIResponsesClient`: the Responses API on Azure OpenAI
//! (`POST {endpoint}/openai/v1/responses`). Like the plain OpenAI Responses
//! client, a reply's `conversation_id` (the Responses API's
//! `previous_response_id`) can be fed into a follow-up call so the service
//! remembers the prior turn without resending history.
//!
//! Skips gracefully unless `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`,
//! and `AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME` are all set. Optional:
//! `AZURE_OPENAI_API_VERSION` (default `"preview"`, the only value Azure
//! currently documents for this surface) and `AZURE_OPENAI_BASE_URL`.
//!
//! ```bash
//! AZURE_OPENAI_ENDPOINT=https://my-resource.openai.azure.com \
//! AZURE_OPENAI_API_KEY=... \
//! AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME=my-gpt4o-deployment \
//! cargo run -p agent-framework-examples --example azure_openai_responses
//! ```

use agent_framework::azure::AzureOpenAIResponsesClient;
use agent_framework::prelude::*;

const REQUIRED_VARS: &[&str] = &[
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME",
];

#[tokio::main]
async fn main() -> Result<()> {
    if let Some(missing) = REQUIRED_VARS.iter().find(|v| std::env::var(v).is_err()) {
        println!("set {missing} (see the other AZURE_OPENAI_* vars in this file's header) to run this example");
        return Ok(());
    }

    let client = AzureOpenAIResponsesClient::from_env()?;

    // First turn: no conversation_id yet.
    let first = client
        .get_response(
            vec![ChatMessage::user("My name is Ada. Remember that.")],
            ChatOptions::new(),
        )
        .await?;
    println!("assistant: {}", first.text());

    // Reuse the returned conversation_id (mapped to `previous_response_id`
    // on the wire) so the service recalls this turn without us resending it.
    let mut options = ChatOptions::new();
    options.conversation_id = first.conversation_id.clone();

    let second = client
        .get_response(vec![ChatMessage::user("What is my name?")], options)
        .await?;
    println!("assistant: {}", second.text());

    Ok(())
}
