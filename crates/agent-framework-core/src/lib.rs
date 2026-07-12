//! # agent-framework-core
//!
//! Core abstractions for `agent-framework-rs`, a Rust implementation of the
//! Microsoft Agent Framework. This crate provides the building blocks:
//!
//! - [`types`] — the data model: messages, content, responses, options.
//! - [`client`] — the [`ChatClient`](client::ChatClient) trait and the
//!   automatic function-invocation loop.
//! - [`agent`] — the [`Agent`](agent::Agent) trait and
//!   [`ChatAgent`](agent::ChatAgent).
//! - [`tools`] — executable tools and hosted-tool markers.
//! - [`threads`] — conversation threads and message stores.
//! - [`memory`] — context / memory providers.
//! - [`middleware`] — agent, chat, and function middleware pipelines.
//! - [`observability`] — OpenTelemetry GenAI-style `tracing` instrumentation.
//! - [`settings`] — secret-masking [`SecretString`](settings::SecretString)
//!   and precedence-based setting resolution.
//! - [`workflow`] — graph-based multi-agent workflow orchestration.
//!
//! ## Example
//!
//! ```no_run
//! use agent_framework_core::prelude::*;
//! # async fn demo(client: impl ChatClient + 'static) -> Result<()> {
//! let agent = ChatAgent::builder(client)
//!     .name("assistant")
//!     .instructions("You are a helpful assistant.")
//!     .build();
//!
//! let response = agent.run_once("Hello!").await?;
//! println!("{}", response.text());
//! # Ok(())
//! # }
//! ```

pub mod agent;
pub mod client;
pub mod error;
pub mod memory;
pub mod middleware;
pub mod observability;
pub mod settings;
pub mod streaming;
pub mod threads;
pub mod tools;
pub mod types;
pub mod workflow;

pub use error::{Error, Result};

/// Commonly used imports.
pub mod prelude {
    pub use crate::agent::{
        Agent, AgentRunOptions, AgentRunStream, AsToolOptions, ChatAgent, ChatAgentBuilder,
    };
    pub use crate::client::{
        ChatClient, ChatStream, FunctionInvokingChatClient, RetryOn, RetryPolicy,
        RetryingChatClient,
    };
    pub use crate::error::{Error, Result};
    pub use crate::memory::{AggregateContextProvider, Context, ContextProvider};
    pub use crate::middleware::{
        AgentContext, ChatContext, FunctionInvocationContext, Middleware, MiddlewarePipeline, Next,
    };
    pub use crate::observability::{ObservabilityConfig, ObservableChatClient};
    pub use crate::settings::{load_setting, SecretString};
    pub use crate::threads::{AgentThread, ChatMessageStore, InMemoryChatMessageStore};
    pub use crate::tools::{
        hosted_code_interpreter, hosted_file_search, hosted_mcp, hosted_web_search, ApprovalMode,
        FunctionInvocationConfig, FunctionTool, McpApprovalMode, Tool, ToolDefinition, ToolKind,
        ToolSource,
    };
    pub use crate::types::{
        AgentResponse, AgentResponseUpdate, ChatMessage, ChatOptions, ChatResponse,
        ChatResponseUpdate, Content, FinishReason, FunctionApprovalRequestContent,
        FunctionApprovalResponseContent, FunctionCallContent, FunctionResultContent,
        ResponseFormat, Role, TextContent, ToolMode, UsageDetails,
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
