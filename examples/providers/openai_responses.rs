//! The OpenAI Responses API client, including reusing a `conversation_id`
//! (the Responses API's `previous_response_id`) across two calls so the
//! service remembers the prior turn without resending history.
//!
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run -p agent-framework-examples --example openai_responses
//! ```

use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIResponsesClient::from_env("gpt-4o-mini")?;

    // First turn: no conversation_id yet.
    let first = client
        .get_response(
            vec![Message::user("My name is Ada. Remember that.")],
            ChatOptions::new(),
        )
        .await?;
    println!("assistant: {}", first.text());

    // The response carries a `conversation_id` (the Responses API's
    // `response.id`, echoed back unless `ChatOptions::store` is explicitly
    // `Some(false)`). Feed it into the next call's options -- the client maps
    // it to `previous_response_id` on the wire -- so the service recalls this
    // turn without us resending the first message.
    let mut options = ChatOptions::new();
    options.conversation_id = first.conversation_id.clone();

    let second = client
        .get_response(vec![Message::user("What is my name?")], options)
        .await?;
    println!("assistant: {}", second.text());

    Ok(())
}
