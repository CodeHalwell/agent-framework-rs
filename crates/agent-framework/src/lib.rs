//! # agent-framework
//!
//! A Rust implementation of the [Microsoft Agent Framework](https://github.com/microsoft/agent-framework)
//! for building AI agents and multi-agent workflows.
//!
//! This is the umbrella crate: it re-exports [`agent_framework_core`] and, via
//! cargo features, the providers:
//!
//! | Feature | Crate | Default |
//! | --- | --- | --- |
//! | `openai` | [`agent_framework_openai`] — OpenAI Chat Completions + Responses API | yes |
//! | `anthropic` | [`agent_framework_anthropic`] — Anthropic (Claude) Messages API | no |
//! | `azure` | [`agent_framework_azure`] — Azure OpenAI (api-key / Entra ID) | no |
//! | `mcp` | [`agent_framework_mcp`] — Model Context Protocol tools | no |
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

#![doc(html_root_url = "https://docs.rs/agent-framework")]

pub use agent_framework_core::*;

/// The OpenAI provider (enabled by the default `openai` feature).
#[cfg(feature = "openai")]
pub use agent_framework_openai as openai;

/// The Anthropic (Claude) provider (enable the `anthropic` feature).
#[cfg(feature = "anthropic")]
pub use agent_framework_anthropic as anthropic;

/// The Azure OpenAI provider (enable the `azure` feature).
#[cfg(feature = "azure")]
pub use agent_framework_azure as azure;

/// Model Context Protocol tools (enable the `mcp` feature).
#[cfg(feature = "mcp")]
pub use agent_framework_mcp as mcp;

/// Commonly used imports for building agents and workflows.
pub mod prelude {
    pub use agent_framework_core::prelude::*;

    #[cfg(feature = "openai")]
    pub use agent_framework_openai::{OpenAIClient, OpenAIResponsesClient};

    #[cfg(feature = "anthropic")]
    pub use agent_framework_anthropic::AnthropicClient;

    #[cfg(feature = "azure")]
    pub use agent_framework_azure::{AzureOpenAIClient, StaticTokenCredential, TokenCredential};

    #[cfg(feature = "mcp")]
    pub use agent_framework_mcp::{McpStdioTool, McpStreamableHttpTool};
}
