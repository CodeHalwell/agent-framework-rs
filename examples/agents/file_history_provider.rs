//! `FileHistoryProvider`: conversation history persisted to a JSON file on
//! disk. Like `InMemoryHistoryProvider` it is a `HistoryProvider` (a
//! `ContextProvider`): `before_run` prepends the stored messages ahead of the
//! run's input and `after_run` records the exchange -- but here every
//! successful run also rewrites the backing file, and constructing a provider
//! on an existing path reloads whatever it contains. That makes a
//! conversation survive a process exit with no explicit save/restore step.
//!
//! This example simulates two process runs in one binary: "run 1" converses
//! and is dropped entirely; "run 2" rebuilds agent, provider, and session
//! from nothing but the file path and continues the conversation. (For
//! explicitly serializing history + session state to strings instead, see
//! `thread_persistence` and `suspend_resume_session`.)
//!
//! Runs fully offline against a canned client -- no API key or network
//! needed.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example file_history_provider
//! ```

use std::path::Path;
use std::sync::Arc;

use agent_framework::prelude::*;
use async_trait::async_trait;

/// Reports how many messages (history + new input) each request carried, so
/// the reloaded history is observable in the output.
#[derive(Clone)]
struct CannedClient;

#[async_trait]
impl ChatClient for CannedClient {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        Ok(ChatResponse::from_text(format!(
            "(canned reply) this request carried {} message(s) of history/context.",
            messages.len()
        )))
    }

    async fn get_streaming_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatStream> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

/// One "process run": build everything from scratch around `path`, ask one
/// question, and drop it all again.
async fn process_run(path: &Path, question: &str) -> Result<()> {
    // Reloads any history already in the file (a missing file starts empty).
    let history = Arc::new(FileHistoryProvider::new(path)?);
    println!(
        "loaded {} message(s) from {}",
        history.list_messages().len(),
        path.display()
    );

    let agent = Agent::builder(CannedClient).name("assistant").build();
    // Attaching the file-backed provider explicitly stops the agent from
    // auto-attaching its own (in-memory) history provider.
    let mut session = AgentSession::new()
        .with_context_providers(vec![history.clone() as Arc<dyn ContextProvider>]);

    let response = agent
        .run(vec![Message::user(question)], Some(&mut session))
        .await?;
    println!("user: {question}\nassistant: {}", response.text());
    println!(
        "file now holds {} message(s)\n",
        history.list_messages().len()
    );
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let path = std::env::temp_dir().join(format!(
        "agent-framework-file-history-{}.json",
        std::process::id()
    ));
    // Start from a clean slate even if a previous run left the file behind.
    let _ = std::fs::remove_file(&path);

    println!("-- process run 1 --");
    process_run(&path, "My favorite color is teal.").await?;

    // Everything from run 1 is gone; only the file remains.
    println!("-- process run 2 (fresh agent + provider, same file) --");
    process_run(&path, "What's my favorite color?").await?;
    println!(
        "(3 messages on the second request -- 2 reloaded from disk plus the new\n\
         question -- confirms the conversation survived the \"restart\".)"
    );

    // Peek at the wire format: a single {"messages": [...]} JSON document.
    let raw = std::fs::read_to_string(&path).map_err(Error::other)?;
    let doc: serde_json::Value = serde_json::from_str(&raw)?;
    println!(
        "\nhistory file is plain JSON with {} message(s) under \"messages\"",
        doc["messages"].as_array().map(Vec::len).unwrap_or(0)
    );

    std::fs::remove_file(&path).ok();
    Ok(())
}
