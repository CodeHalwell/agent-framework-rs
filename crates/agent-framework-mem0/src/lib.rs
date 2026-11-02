//! # agent-framework-mem0
//!
//! A [`ContextProvider`](agent_framework_core::memory::ContextProvider)
//! backed by the hosted [Mem0](https://mem0.ai) memory API, porting
//! `agent_framework_mem0` from the Python Agent Framework.
//!
//! See the [`Mem0Provider`] docs for the exact REST contract this crate
//! targets (`/v1/memories/` for writes, `/v2/memories/search/` for reads)
//! and how/why it differs from the Python package, which delegates to the
//! `mem0` Python SDK rather than making HTTP calls directly.
//!
//! ```no_run
//! use agent_framework_core::prelude::*;
//! use agent_framework_mem0::Mem0Provider;
//! use std::sync::Arc;
//!
//! # async fn demo(client: impl ChatClient + 'static) -> Result<()> {
//! // Reads MEM0_API_KEY (and optional MEM0_API_BASE) from the environment.
//! let memory = Mem0Provider::from_env()?.with_user_id("user-42");
//!
//! let agent = Agent::builder(client)
//!     .instructions("You are a helpful assistant.")
//!     .context_provider(Arc::new(memory))
//!     .build();
//!
//! let response = agent.run_once("What do you remember about me?").await?;
//! println!("{}", response.text());
//! # Ok(())
//! # }
//! ```

pub mod provider;

pub use provider::{Mem0Provider, ADD_PATH, DEFAULT_API_BASE, DEFAULT_CONTEXT_PROMPT, SEARCH_PATH};

/// This crate's HTTP client, sending the framework `User-Agent`
/// ([`agent_framework_core::user_agent::user_agent`]) so providers can
/// attribute the traffic. A static ASCII token cannot fail to become a
/// header; the fallback only keeps this infallible.
pub(crate) fn http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent(agent_framework_core::user_agent::user_agent(
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION"),
        ))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
}
