//! Automatic history compaction inside an agent:
//! `AgentBuilder::with_compaction(strategy)` attaches a `CompactionProvider`
//! (a `ContextProvider`) as one of the agent's own providers. Agent-owned
//! providers always run *after* the session's -- including the auto-attached
//! `InMemoryHistoryProvider` -- so on every run compaction sees the full,
//! history-prepended message list and shrinks it *before it reaches the
//! model*. The stored history itself is untouched: only the outgoing request
//! is compacted.
//!
//! Two agents run the same five-turn conversation against a client that
//! records how many messages each request carried: without compaction the
//! request grows every turn; with a `SlidingWindow(4)` it plateaus at 5
//! messages -- the 4 retained history messages plus the current turn's
//! input, which is appended after the providers run and is never compacted
//! away. See `compaction_basics` for the strategies themselves.
//!
//! Runs fully offline against a counting client -- no API key or network
//! needed.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example compaction_provider
//! ```

use std::sync::{Arc, Mutex};

use agent_framework::prelude::*;
use async_trait::async_trait;

/// Records how many messages each request carried, standing in for a real
/// model so the effect of compaction on the outgoing request is observable.
#[derive(Clone, Default)]
struct CountingClient {
    sizes: Arc<Mutex<Vec<usize>>>,
}

#[async_trait]
impl ChatClient for CountingClient {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        self.sizes.lock().unwrap().push(messages.len());
        Ok(ChatResponse::from_text(format!(
            "(canned reply to a {}-message request)",
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

/// Run the same five turns against `agent`, returning per-turn request sizes.
async fn five_turns(agent: &Agent, sizes: &Arc<Mutex<Vec<usize>>>) -> Result<Vec<usize>> {
    // `create_session` attaches an `InMemoryHistoryProvider`, so history
    // accumulates across turns exactly as in any multi-turn conversation.
    let mut session = agent.create_session();
    for turn in 1..=5 {
        agent
            .run(
                vec![Message::user(format!("question #{turn}"))],
                Some(&mut session),
            )
            .await?;
    }
    Ok(sizes.lock().unwrap().clone())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Baseline: no compaction -- every run sends the whole history.
    let plain_client = CountingClient::default();
    let plain_sizes = plain_client.sizes.clone();
    let plain = Agent::builder(plain_client).name("plain").build();
    let plain_counts = five_turns(&plain, &plain_sizes).await?;

    // Compacting: keep only the last 4 (non-system) history messages per
    // request; the current turn's input always rides on top.
    let compact_client = CountingClient::default();
    let compact_sizes = compact_client.sizes.clone();
    let compacting = Agent::builder(compact_client)
        .name("compacting")
        .with_compaction(SlidingWindow::new(4))
        .build();
    let compact_counts = five_turns(&compacting, &compact_sizes).await?;

    println!("messages sent to the model per turn (history + this turn's input):\n");
    println!("  turn | without compaction | with SlidingWindow(4)");
    println!("  -----+--------------------+----------------------");
    for turn in 0..5 {
        println!(
            "    {}  | {:>18} | {:>21}",
            turn + 1,
            plain_counts[turn],
            compact_counts[turn]
        );
    }

    assert_eq!(plain_counts, vec![1, 3, 5, 7, 9], "grows by 2 every turn");
    assert_eq!(
        compact_counts,
        vec![1, 3, 5, 5, 5],
        "plateaus at window (4 history) + 1 input"
    );

    println!(
        "\nWithout compaction each turn adds its question + answer to the request;\n\
         with compaction it plateaus at the window (4 history messages) plus the\n\
         current turn's input, while the session's stored history keeps the full\n\
         conversation."
    );
    Ok(())
}
