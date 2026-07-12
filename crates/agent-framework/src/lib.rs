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
//! | `ollama` | [`agent_framework_ollama`] — Ollama (local/remote, OpenAI-compatible) | no |
//! | `gemini` | [`agent_framework_gemini`] — Google Gemini `generateContent` API | no |
//! | `mistral` | [`agent_framework_mistral`] — Mistral AI Chat Completions API | no |
//! | `azure` | [`agent_framework_azure`] — Azure OpenAI (api-key / Entra ID) | no |
//! | `mcp` | [`agent_framework_mcp`] — Model Context Protocol tools (stdio, HTTP, websocket) | no |
//! | `a2a` | [`agent_framework_a2a`] — Agent2Agent protocol client | no |
//! | `declarative` | [`agent_framework_declarative`] — YAML/JSON agents & workflows | no |
//! | `hosting` | [`agent_framework_hosting`] — serve agents over HTTP (DevUI-style, A2A, OpenAI-compatible) | no |
//! | `redis` | [`agent_framework_redis`] — Redis chat-message store & context provider | no |
//! | `mem0` | [`agent_framework_mem0`] — Mem0 long-term memory provider | no |
//! | `azure-ai` | [`agent_framework_azure_ai`] — Azure AI Foundry persistent agents | no |
//! | `azure-ai-search` | [`agent_framework_azure_ai_search`] — Azure AI Search memory | no |
//! | `cosmos` | [`agent_framework_cosmos`] — Cosmos DB NoSQL message store | no |
//! | `copilotstudio` | [`agent_framework_copilotstudio`] — Copilot Studio agents | no |
//! | `purview` | [`agent_framework_purview`] — Purview compliance middleware | no |
//!
//! `full` enables everything.
//!
//! ```no_run
//! use agent_framework::prelude::*;
//!
//! # async fn demo() -> Result<()> {
//! let client = OpenAIChatCompletionClient::from_env("gpt-4o-mini")?;
//! let agent = Agent::builder(client)
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

/// The Ollama provider (enable the `ollama` feature).
#[cfg(feature = "ollama")]
pub use agent_framework_ollama as ollama;

/// The Google Gemini provider (enable the `gemini` feature).
#[cfg(feature = "gemini")]
pub use agent_framework_gemini as gemini;

/// The Mistral AI provider (enable the `mistral` feature).
#[cfg(feature = "mistral")]
pub use agent_framework_mistral as mistral;

/// The Azure OpenAI provider (enable the `azure` feature).
#[cfg(feature = "azure")]
pub use agent_framework_azure as azure;

/// Model Context Protocol tools (enable the `mcp` feature).
#[cfg(feature = "mcp")]
pub use agent_framework_mcp as mcp;

/// Agent2Agent protocol client (enable the `a2a` feature).
#[cfg(feature = "a2a")]
pub use agent_framework_a2a as a2a;

/// Declarative YAML/JSON agents and workflows (enable the `declarative` feature).
#[cfg(feature = "declarative")]
pub use agent_framework_declarative as declarative;

/// HTTP hosting: DevUI-style, A2A, and OpenAI-compatible serving (enable the `hosting` feature).
#[cfg(feature = "hosting")]
pub use agent_framework_hosting as hosting;

/// Redis-backed chat-message store and context provider (enable the `redis` feature).
#[cfg(feature = "redis")]
pub use agent_framework_redis as redis;

/// Mem0 long-term memory context provider (enable the `mem0` feature).
#[cfg(feature = "mem0")]
pub use agent_framework_mem0 as mem0;

/// Azure AI Foundry persistent-agents client (enable the `azure-ai` feature).
#[cfg(feature = "azure-ai")]
pub use agent_framework_azure_ai as azure_ai;

/// Azure AI Search context provider (enable the `azure-ai-search` feature).
#[cfg(feature = "azure-ai-search")]
pub use agent_framework_azure_ai_search as azure_ai_search;

/// Azure Cosmos DB NoSQL chat-message store (enable the `cosmos` feature).
#[cfg(feature = "cosmos")]
pub use agent_framework_cosmos as cosmos;

/// Microsoft Copilot Studio agent client (enable the `copilotstudio` feature).
#[cfg(feature = "copilotstudio")]
pub use agent_framework_copilotstudio as copilotstudio;

/// Microsoft Purview compliance middleware (enable the `purview` feature).
#[cfg(feature = "purview")]
pub use agent_framework_purview as purview;

/// Commonly used imports for building agents and workflows.
pub mod prelude {
    pub use agent_framework_core::prelude::*;

    #[cfg(feature = "openai")]
    pub use agent_framework_openai::{OpenAIChatClient, OpenAIChatCompletionClient};

    #[cfg(feature = "anthropic")]
    pub use agent_framework_anthropic::AnthropicClient;

    #[cfg(feature = "ollama")]
    pub use agent_framework_ollama::OllamaChatClient;

    #[cfg(feature = "gemini")]
    pub use agent_framework_gemini::GeminiChatClient;

    #[cfg(feature = "mistral")]
    pub use agent_framework_mistral::MistralChatClient;

    #[cfg(feature = "azure")]
    pub use agent_framework_azure::{AzureOpenAIClient, StaticTokenCredential, TokenCredential};

    #[cfg(feature = "mcp")]
    pub use agent_framework_mcp::{McpStdioTool, McpStreamableHttpTool, McpWebsocketTool};

    #[cfg(feature = "a2a")]
    pub use agent_framework_a2a::{A2AAgent, A2AClient};

    #[cfg(feature = "declarative")]
    pub use agent_framework_declarative::DeclarativeLoader;

    #[cfg(feature = "hosting")]
    pub use agent_framework_hosting::AgentHost;

    #[cfg(feature = "redis")]
    pub use agent_framework_redis::{RedisChatMessageStore, RedisContextProvider};

    #[cfg(feature = "mem0")]
    pub use agent_framework_mem0::Mem0Provider;

    #[cfg(feature = "azure-ai")]
    pub use agent_framework_azure_ai::AzureAIAgentClient;

    #[cfg(feature = "azure-ai-search")]
    pub use agent_framework_azure_ai_search::AzureAISearchProvider;

    #[cfg(feature = "cosmos")]
    pub use agent_framework_cosmos::CosmosChatMessageStore;

    #[cfg(feature = "copilotstudio")]
    pub use agent_framework_copilotstudio::CopilotStudioAgent;

    #[cfg(feature = "purview")]
    pub use agent_framework_purview::{PurviewAgentMiddleware, PurviewChatMiddleware};
}
