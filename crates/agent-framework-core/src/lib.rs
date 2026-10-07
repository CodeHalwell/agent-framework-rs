//! # agent-framework-core
//!
//! Core abstractions for `agent-framework-rs`, a Rust implementation of the
//! Microsoft Agent Framework. This crate provides the building blocks:
//!
//! - [`types`] — the data model: messages, content, responses, options.
//! - [`client`] — the [`ChatClient`](client::ChatClient) trait and the
//!   automatic function-invocation loop.
//! - [`agent`] — the [`SupportsAgentRun`](agent::SupportsAgentRun) trait and
//!   [`Agent`](agent::Agent).
//! - [`compaction`] — conversation-history compaction strategies and the
//!   [`Tokenizer`](compaction::Tokenizer) abstraction.
//! - [`tools`] — executable tools and hosted-tool markers.
//! - [`session`] — [`AgentSession`](session::AgentSession), a lightweight
//!   conversation identity + state container.
//! - `harness` (feature `experimental-harness`) — the experimental agent
//!   harness: an agent loop, a todo list, standing tool approvals and
//!   operating modes.
//! - [`history`] — [`HistoryProvider`](history::HistoryProvider)s: conversation
//!   history as a [`ContextProvider`](memory::ContextProvider).
//! - [`memory`] — context / memory providers.
//! - [`middleware`] — agent, chat, and function middleware pipelines.
//! - [`observability`] — OpenTelemetry GenAI-style `tracing` instrumentation.
//! - [`skills`] — [`Skill`](skills::Skill) capability packages, surfaced via
//!   [`SkillsProvider`](skills::SkillsProvider), a
//!   [`ContextProvider`](memory::ContextProvider) that progressively
//!   discloses skill instructions and resources through
//!   framework-generated tools.
//! - [`settings`] — secret-masking [`SecretString`](settings::SecretString)
//!   and precedence-based setting resolution.
//! - [`workflow`] — graph-based multi-agent workflow orchestration.
//!
//! ## Experimental features
//!
//! APIs that upstream still marks experimental sit behind `experimental-*`
//! cargo features and carry no semver promise: `experimental-vector-stores`
//! (the `vectors` module), `experimental-file-history`
//! (`history::FileHistoryProvider`), `experimental-progressive-tools`
//! (`middleware::LiveToolList`) and `experimental-harness` (the `harness`
//! module). See `docs/feature-stages.md` in the repository.
//!
//! ## Example
//!
//! ```no_run
//! use agent_framework_core::prelude::*;
//! # async fn demo(client: impl ChatClient + 'static) -> Result<()> {
//! let agent = Agent::builder(client)
//!     .name("assistant")
//!     .instructions("You are a helpful assistant.")
//!     .build();
//!
//! let response = agent.run_once("Hello!").await?;
//! println!("{}", response.text());
//! # Ok(())
//! # }
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]

/// Declares an item `pub` when its experimental cargo feature is on and
/// `pub(crate)` otherwise, for items the crate uses internally either way.
/// Items nothing else uses are gated with a plain `#[cfg(feature = ...)]`.
macro_rules! experimental {
    ($feature:literal, $(#[$meta:meta])* pub $($item:tt)*) => {
        #[cfg(feature = $feature)]
        #[cfg_attr(docsrs, doc(cfg(feature = $feature)))]
        $(#[$meta])*
        pub $($item)*

        #[cfg(not(feature = $feature))]
        #[allow(dead_code)]
        $(#[$meta])*
        pub(crate) $($item)*
    };
}

pub mod agent;
pub mod client;
pub mod compaction;
pub mod error;
#[cfg(feature = "experimental-harness")]
#[cfg_attr(docsrs, doc(cfg(feature = "experimental-harness")))]
pub mod harness;
pub mod history;
pub mod memory;
pub mod middleware;
pub mod observability;
pub mod session;
pub mod settings;
pub mod skills;
pub mod storage_keys;
pub mod streaming;
pub mod tools;
pub mod types;
#[cfg(feature = "experimental-vector-stores")]
#[cfg_attr(docsrs, doc(cfg(feature = "experimental-vector-stores")))]
pub mod vectors;
pub mod workflow;

pub use error::{Error, Result};

/// Commonly used imports.
pub mod prelude {
    pub use crate::agent::{
        Agent, AgentBuilder, AgentRunOptions, AgentRunStream, AgentToolStreamCallback,
        AsToolOptions, SupportsAgentRun,
    };
    pub use crate::client::{
        ChatClient, ChatStream, EmbeddingClient, FunctionInvokingChatClient, RetryOn, RetryPolicy,
        RetryingChatClient,
    };
    pub use crate::compaction::{
        compact, ApproxTokenizer, CompactionProvider, CompactionStrategy, SelectiveToolResult,
        SlidingWindow, TokenBudget, Tokenizer, Truncation,
    };
    pub use crate::error::{Error, Result};
    #[cfg(feature = "experimental-file-history")]
    pub use crate::history::FileHistoryProvider;
    pub use crate::history::{HistoryProvider, InMemoryHistoryProvider};
    pub use crate::memory::{ContextProvider, SessionContext};
    #[cfg(feature = "experimental-progressive-tools")]
    pub use crate::middleware::LiveToolList;
    pub use crate::middleware::{
        AgentContext, ChatContext, FunctionInvocationContext, Middleware, MiddlewarePipeline, Next,
    };
    pub use crate::observability::{ObservabilityConfig, ObservableChatClient};
    pub use crate::session::{AgentSession, SessionState};
    pub use crate::settings::{load_setting, SecretString};
    pub use crate::skills::{Skill, SkillsProvider};
    pub use crate::tools::{
        hosted_code_interpreter, hosted_file_search, hosted_image_generation, hosted_mcp,
        hosted_web_search, ApprovalMode, FunctionInvocationConfig, FunctionTool, McpApprovalMode,
        Tool, ToolDefinition, ToolKind, ToolSource,
    };
    pub use crate::types::{
        AgentResponse, AgentResponseUpdate, ChatOptions, ChatResponse, ChatResponseUpdate, Content,
        Embedding, EmbeddingGenerationOptions, EmbeddingInput, FinishReason,
        FunctionApprovalRequestContent, FunctionApprovalResponseContent, FunctionCallContent,
        FunctionResultContent, GeneratedEmbeddings, Message, ResponseFormat, Role, TextContent,
        ToolMode, UsageDetails,
    };
    pub use crate::workflow::{
        CheckpointStorage, ConcurrentBuilder, Executor, FileCheckpointStorage, GroupChatBuilder,
        GroupChatDirective, GroupChatManager, GroupChatState, HandoffBuilder,
        HandoffInteractionMode, InMemoryCheckpointStorage, MagenticBuilder, MagenticContext,
        MagenticManager, MagenticPlanReviewDecision, MagenticPlanReviewRequest,
        MagenticStallInterventionDecision, MagenticStallInterventionRequest, RequestInfoExecutor,
        SequentialBuilder, SharedState, StandardMagenticManager, Workflow, WorkflowAgent,
        WorkflowAgentExt, WorkflowBuilder, WorkflowContext, WorkflowEvent, WorkflowExecutor,
        WorkflowRun, WorkflowRunState,
    };
}
