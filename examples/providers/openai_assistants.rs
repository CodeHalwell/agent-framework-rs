//! `OpenAIAssistantsClient`: the OpenAI Assistants (beta) API. Unlike Chat
//! Completions/Responses, this API is thread-based server-side: a fresh
//! assistant (an "agent" in the Assistants API's own terminology) is created
//! lazily on first use, and each reply carries a `conversation_id` (the
//! Assistants `thread_id`) that a follow-up call can reuse to continue the
//! same server-side thread.
//!
//! Skips gracefully unless `OPENAI_API_KEY` is set.
//!
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run -p agent-framework-examples --example openai_assistants
//! ```

use agent_framework::openai::OpenAIAssistantsClient;
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let Ok(_) = std::env::var("OPENAI_API_KEY") else {
        println!("set OPENAI_API_KEY to run this example");
        return Ok(());
    };

    // With no `with_assistant_id`, a transient assistant is auto-created
    // from `model` on first use; `close()` at the end deletes it.
    let client = OpenAIAssistantsClient::from_env("gpt-4o-mini")?
        .with_assistant_name("rust-example-assistant");

    // First run: no conversation_id yet -- a new Assistants thread is
    // created.
    let first = client
        .get_response(
            vec![ChatMessage::user("My name is Ada. Remember that.")],
            ChatOptions::new(),
        )
        .await?;
    println!("assistant: {}", first.text());

    // Second run: reuse the conversation_id (the Assistants thread id) the
    // first response returned, so the service continues the same thread
    // instead of starting a fresh one.
    let mut options = ChatOptions::new();
    options.conversation_id = first.conversation_id.clone();

    let second = client
        .get_response(vec![ChatMessage::user("What is my name?")], options)
        .await?;
    println!("assistant: {}", second.text());

    // Delete the transient assistant this client auto-created.
    client.close().await?;
    println!("(transient assistant deleted)");

    Ok(())
}
