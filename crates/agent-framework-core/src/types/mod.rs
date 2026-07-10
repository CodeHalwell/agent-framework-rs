//! Core data model: content, messages, responses, and options.
//!
//! This is the Rust equivalent of `agent_framework._types`.

mod content;
mod message;
mod options;
mod response;

pub use content::{
    prepare_function_call_results, CitationAnnotation, Content, DataContent, ErrorContent,
    FunctionApprovalRequestContent, FunctionApprovalResponseContent, FunctionArguments,
    FunctionCallContent, FunctionResultContent, HostedFileContent, HostedVectorStoreContent,
    TextContent, TextReasoningContent, TextSpanRegion, UriContent, UsageContent, UsageDetails,
};
pub use message::{prepare_messages, ChatMessage, IntoMessages, Role};
pub use options::{ChatOptions, ResponseFormat, ToolMode};
pub use response::{
    AgentRunResponse, AgentRunResponseUpdate, ChatResponse, ChatResponseUpdate, FinishReason,
};
