//! Talk to a remote Agent2Agent (A2A) server as if it were a local agent:
//! `A2AAgent` implements the `Agent` trait over the A2A JSON-RPC protocol,
//! and reusing one `AgentThread` carries the remote `contextId`/`taskId`
//! across turns for a real multi-turn conversation.
//!
//! Point it at any A2A-compliant server -- for instance the one from the
//! `hosting_server` example (`--features hosting`, nested A2A router), or a
//! public sample server:
//! ```bash
//! A2A_AGENT_URL=http://127.0.0.1:8080/a2a \
//! cargo run -p agent-framework-examples --example a2a_client
//! ```

use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let Ok(url) = std::env::var("A2A_AGENT_URL") else {
        println!("set A2A_AGENT_URL to an A2A server's endpoint to run this example");
        return Ok(());
    };

    let agent = A2AAgent::from_url("remote-agent", &url);

    // Optional: fetch the server's agent card (name, description, skills)
    // from its .well-known discovery document.
    match agent.initialize().await {
        Ok(card) => println!("connected to '{}': {}", card.name, card.description),
        Err(e) => println!("no agent card ({e}); proceeding with bare JSON-RPC"),
    }

    // Single-shot, stateless call.
    let response = agent.run_once("Hello! What can you do?").await?;
    println!("agent: {}", response.text());

    // Multi-turn: reuse a thread so the remote contextId/taskId are replayed
    // on the next call and the server sees one continuous conversation.
    let mut thread = agent.get_new_thread();
    agent
        .run(
            vec![ChatMessage::user("My name is Ada.")],
            Some(&mut thread),
        )
        .await?;
    let reply = agent
        .run(
            vec![ChatMessage::user("What is my name?")],
            Some(&mut thread),
        )
        .await?;
    println!("agent (same conversation): {}", reply.text());

    Ok(())
}
