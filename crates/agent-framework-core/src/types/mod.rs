//! Core data model: content, messages, responses, and options.
//!
//! This is the Rust equivalent of `agent_framework._types`.

mod content;
mod message;
mod options;
mod response;

pub use content::{
    prepare_function_call_results, Annotation, AnnotationKind, Content, DataContent, ErrorContent,
    FunctionApprovalRequestContent, FunctionApprovalResponseContent, FunctionArguments,
    FunctionCallContent, FunctionResultContent, HostedFileContent, HostedVectorStoreContent,
    TextContent, TextReasoningContent, TextSpanRegion, UriContent, UsageContent, UsageDetails,
};
pub use message::{prepare_messages, IntoMessages, Message, Role};
pub use options::{ChatOptions, ResponseFormat, ToolMode};
pub use response::{
    AgentResponse, AgentResponseUpdate, ChatResponse, ChatResponseUpdate, ContinuationToken,
    FinishReason,
};
