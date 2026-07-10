//! # agent-framework
//!
//! A Rust implementation of the [Microsoft Agent Framework](https://github.com/microsoft/agent-framework)
//! for building AI agents and multi-agent workflows.
//!
//! This is the umbrella crate: it re-exports [`agent_framework_core`] and, with
//! the default `openai` feature, the OpenAI provider.
//!
//! ```no_run
//! use agent_framework::prelude::*;
//!
//! # async fn demo() -> Result<()> {
//! let client = OpenAIClient::from_env("gpt-4o-mini")?;
//! let agent = ChatAgent::builder(client)
//!     .name("assistant")
//!     .instructions("You are a helpful assistant.")
//!     .build();
//!
//! let response = agent.run_once("What is the capital of France?").await?;
//! println!("{}", response.text());
//! # Ok(())
//! # }
//! ```
//!
//! ## Feature parity roadmap
//!
//! The crate mirrors the architecture of the Python/.NET framework. Implemented:
//! the core data model, chat-client abstraction with an automatic
//! function-invocation loop, agents, tools, threads, memory/context providers,
//! middleware pipelines, a graph-based workflow engine, and sequential /
//! concurrent orchestration. See the repository README for the full status and
//! roadmap toward complete parity (additional providers, group-chat / handoff /
//! magentic orchestration, checkpointing, DevUI, and declarative agents).

#![doc(html_root_url = "https://docs.rs/agent-framework")]

pub use agent_framework_core::*;

/// The OpenAI provider (enabled by the default `openai` feature).
#[cfg(feature = "openai")]
pub use agent_framework_openai as openai;

/// Commonly used imports for building agents and workflows.
pub mod prelude {
    pub use agent_framework_core::prelude::*;

    #[cfg(feature = "openai")]
    pub use agent_framework_openai::OpenAIClient;
}
