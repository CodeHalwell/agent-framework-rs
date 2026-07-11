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
    ///
    /// Used for non-HTTP service failures (transport errors, stream-decode
    /// errors, in-body error payloads on an otherwise-successful response). For
    /// a non-success HTTP status, prefer [`Error::ServiceStatus`], which also
    /// carries the status code and any `Retry-After`.
    #[error("service error: {0}")]
    Service(String),

    /// A chat client / service returned a non-success HTTP status.
    ///
    /// Distinct from [`Error::Service`] so a retry layer can inspect the
    /// numeric status code and any server-advised `Retry-After` delay (in
    /// seconds). Displays like [`Error::Service`] (a `service error: ...`
    /// message), with the status code folded into the message.
    ///
    /// This is the fallback classification for a non-success status that
    /// isn't one of the more specific variants below (notably `408`/`429`/
    /// `5xx`, which a retry layer treats as transient) — see
    /// [`Error::ServiceInvalidAuth`], [`Error::ServiceInvalidRequest`], and
    /// [`Error::ServiceContentFilter`] for statuses a provider client can
    /// classify more precisely.
    #[error("service error: {message}")]
    ServiceStatus {
        /// The HTTP status code returned by the service.
        status: u16,
        /// A human-readable message (typically the response body).
        message: String,
        /// The server-advised retry delay in seconds, parsed from the
        /// `Retry-After` header when present.
        retry_after: Option<f64>,
    },

    /// The service rejected the request due to missing or invalid
    /// credentials (typically HTTP `401`/`403`).
    ///
    /// Mirrors upstream's `ServiceInvalidAuthError`. Like [`Error::Service`],
    /// this carries no status code of its own (the numeric status, when
    /// known, is folded into the message by the provider client) — it exists
    /// so callers, and the default retry policy, can treat authentication /
    /// authorization failures as definitively non-transient without
    /// inspecting a status code themselves. Never retried by
    /// [`RetryOn::Default`](crate::client::RetryOn::Default).
    #[error("service error: {message}")]
    ServiceInvalidAuth {
        /// A human-readable message (typically the response body).
        message: String,
    },

    /// The service rejected the request as malformed or otherwise invalid
    /// (typically HTTP `400`/`404`/`422`) for a reason other than content
    /// filtering.
    ///
    /// Mirrors upstream's `ServiceInvalidRequestError`. See
    /// [`Error::ServiceContentFilter`] for the content-filter-specific case.
    /// Never retried by [`RetryOn::Default`](crate::client::RetryOn::Default)
    /// — a request that was rejected as invalid will be rejected again
    /// unchanged.
    #[error("service error: {message}")]
    ServiceInvalidRequest {
        /// A human-readable message (typically the response body).
        message: String,
    },

    /// The service refused the request (or part of a response) because it
    /// tripped a content filter / moderation policy.
    ///
    /// Mirrors upstream's `ServiceContentFilterException`
    /// (`OpenAIContentFilterException` for OpenAI/Azure OpenAI specifically).
    /// Never retried by [`RetryOn::Default`](crate::client::RetryOn::Default)
    /// — the content, not the service, is the problem.
    #[error("service error: {message}")]
    ServiceContentFilter {
        /// A human-readable message (typically the response body).
        message: String,
    },

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

    /// Create an [`Error::ServiceStatus`] from an HTTP status code, a message,
    /// and an optional `Retry-After` delay (in seconds).
    pub fn service_status(status: u16, msg: impl fmt::Display, retry_after: Option<f64>) -> Self {
        Error::ServiceStatus {
            status,
            message: msg.to_string(),
            retry_after,
        }
    }

    /// Create an [`Error::ServiceInvalidAuth`] from anything displayable.
    pub fn service_invalid_auth(msg: impl fmt::Display) -> Self {
        Error::ServiceInvalidAuth {
            message: msg.to_string(),
        }
    }

    /// Create an [`Error::ServiceInvalidRequest`] from anything displayable.
    pub fn service_invalid_request(msg: impl fmt::Display) -> Self {
        Error::ServiceInvalidRequest {
            message: msg.to_string(),
        }
    }

    /// Create an [`Error::ServiceContentFilter`] from anything displayable.
    pub fn service_content_filter(msg: impl fmt::Display) -> Self {
        Error::ServiceContentFilter {
            message: msg.to_string(),
        }
    }

    /// The HTTP status code carried by this error, if it is an
    /// [`Error::ServiceStatus`].
    pub fn status(&self) -> Option<u16> {
        match self {
            Error::ServiceStatus { status, .. } => Some(*status),
            _ => None,
        }
    }

    /// The server-advised retry delay in seconds, if this is an
    /// [`Error::ServiceStatus`] that carried a `Retry-After` header.
    pub fn retry_after(&self) -> Option<f64> {
        match self {
            Error::ServiceStatus { retry_after, .. } => *retry_after,
            _ => None,
        }
    }

    /// Create an [`Error::Tool`] from anything displayable.
    pub fn tool(msg: impl fmt::Display) -> Self {
        Error::Tool(msg.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn service_invalid_auth_constructor_and_display() {
        let err = Error::service_invalid_auth("OpenAI API error 401: unauthorized");
        assert!(matches!(err, Error::ServiceInvalidAuth { .. }));
        assert_eq!(
            err.to_string(),
            "service error: OpenAI API error 401: unauthorized"
        );
        // Not a `ServiceStatus`, so it carries no status/retry_after of its
        // own (the numeric status lives in the message text instead).
        assert_eq!(err.status(), None);
        assert_eq!(err.retry_after(), None);
    }

    #[test]
    fn service_invalid_request_constructor_and_display() {
        let err = Error::service_invalid_request("OpenAI API error 400: bad request");
        assert!(matches!(err, Error::ServiceInvalidRequest { .. }));
        assert_eq!(
            err.to_string(),
            "service error: OpenAI API error 400: bad request"
        );
    }

    #[test]
    fn service_content_filter_constructor_and_display() {
        let err = Error::service_content_filter("OpenAI API error 400: content filtered");
        assert!(matches!(err, Error::ServiceContentFilter { .. }));
        assert_eq!(
            err.to_string(),
            "service error: OpenAI API error 400: content filtered"
        );
    }

    /// The new variants must not silently become retryable-looking:
    /// `status()`/`retry_after()` only ever return `Some` for
    /// [`Error::ServiceStatus`].
    #[test]
    fn new_variants_are_not_service_status() {
        for err in [
            Error::service_invalid_auth("x"),
            Error::service_invalid_request("x"),
            Error::service_content_filter("x"),
        ] {
            assert_eq!(err.status(), None, "{err:?}");
            assert_eq!(err.retry_after(), None, "{err:?}");
        }
    }
}
