//! Error types for the agent framework.

use std::fmt;

/// The result type used throughout the framework.
pub type Result<T> = std::result::Result<T, Error>;

/// The primary error type for the agent framework.
///
/// This mirrors the exception hierarchy used by the Python
/// `agent_framework.exceptions` module while remaining idiomatic Rust.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// An error occurred while initializing an agent.
    #[error("agent initialization error: {0}")]
    AgentInitialization(String),

    /// An error occurred while executing an agent run.
    #[error("agent execution error: {0}")]
    AgentExecution(String),

    /// An error occurred while (de)serializing a value.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// A content item could not be parsed or was of an unknown type.
    #[error("content error: {0}")]
    Content(String),

    /// A tool/function invocation failed.
    #[error("tool error: {0}")]
    Tool(String),

    /// A chat client / service returned an error.
    #[error("service error: {0}")]
    Service(String),

    /// A workflow validation or execution error.
    #[error("workflow error: {0}")]
    Workflow(String),

    /// Two streamed content items could not be merged (mismatched ids).
    #[error("addition item mismatch: {0}")]
    AdditionItemMismatch(String),

    /// A required configuration value was missing or invalid.
    #[error("configuration error: {0}")]
    Configuration(String),

    /// An underlying JSON error.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// Any other error, wrapping a boxed source.
    #[error("{0}")]
    Other(String),
}

impl Error {
    /// Create an [`Error::Other`] from anything displayable.
    pub fn other(msg: impl fmt::Display) -> Self {
        Error::Other(msg.to_string())
    }

    /// Create an [`Error::Service`] from anything displayable.
    pub fn service(msg: impl fmt::Display) -> Self {
        Error::Service(msg.to_string())
    }

    /// Create an [`Error::Tool`] from anything displayable.
    pub fn tool(msg: impl fmt::Display) -> Self {
        Error::Tool(msg.to_string())
    }
}
