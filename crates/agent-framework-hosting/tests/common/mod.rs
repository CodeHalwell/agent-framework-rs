//! Shared test helpers: a mock agent, a tiny workflow, and HTTP/SSE utilities
//! for driving routers via `tower::ServiceExt::oneshot` (no sockets).
#![allow(dead_code)]

use std::sync::Arc;

use async_trait::async_trait;
use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use axum::Router;
use futures::StreamExt;
use serde_json::Value;
use tower::ServiceExt;

use agent_framework_core::agent::{Agent, AgentRunOptions, AgentRunStream};
use agent_framework_core::error::Result;
use agent_framework_core::threads::AgentThread;
use agent_framework_core::types::{
    AgentRunResponse, AgentRunResponseUpdate, ChatMessage, Content, Role, UsageDetails,
};
use agent_framework_core::workflow::{FunctionExecutor, Workflow, WorkflowBuilder};

/// A scripted agent that streams multiple text deltas, to exercise the live SSE
/// paths: `run` returns the concatenation as one message; `run_stream` yields
/// one [`AgentRunResponseUpdate`] per delta (real incremental streaming).
pub struct StreamingAgent {
    id: String,
    deltas: Vec<String>,
}

impl StreamingAgent {
    pub fn new(id: impl Into<String>, deltas: Vec<&str>) -> Self {
        Self {
            id: id.into(),
            deltas: deltas.into_iter().map(str::to_string).collect(),
        }
    }

    pub fn arc(self) -> Arc<dyn Agent> {
        Arc::new(self)
    }
}

#[async_trait]
impl Agent for StreamingAgent {
    async fn run(
        &self,
        _messages: Vec<ChatMessage>,
        _thread: Option<&mut AgentThread>,
    ) -> Result<AgentRunResponse> {
        Ok(AgentRunResponse {
            messages: vec![ChatMessage::assistant(self.deltas.concat())],
            ..Default::default()
        })
    }

    async fn run_stream(
        &self,
        _messages: Vec<ChatMessage>,
        _thread: Option<AgentThread>,
        _options: Option<AgentRunOptions>,
    ) -> Result<AgentRunStream> {
        let updates: Vec<Result<AgentRunResponseUpdate>> = self
            .deltas
            .iter()
            .map(|d| {
                Ok(AgentRunResponseUpdate {
                    contents: vec![Content::text(d)],
                    role: Some(Role::assistant()),
                    ..Default::default()
                })
            })
            .collect();
        Ok(futures::stream::iter(updates).boxed())
    }

    fn id(&self) -> &str {
        &self.id
    }
}

/// A scripted agent: echoes the concatenated input text behind a fixed prefix,
/// so tests can verify both routing and input parsing. Optionally reports usage.
pub struct MockAgent {
    id: String,
    name: Option<String>,
    prefix: String,
    usage: Option<UsageDetails>,
}

impl MockAgent {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: None,
            prefix: "echo: ".to_string(),
            usage: None,
        }
    }

    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    pub fn with_usage(mut self, input: u64, output: u64) -> Self {
        self.usage = Some(UsageDetails {
            input_token_count: Some(input),
            output_token_count: Some(output),
            total_token_count: Some(input + output),
            ..Default::default()
        });
        self
    }

    pub fn arc(self) -> Arc<dyn Agent> {
        Arc::new(self)
    }
}

#[async_trait]
impl Agent for MockAgent {
    async fn run(
        &self,
        messages: Vec<ChatMessage>,
        _thread: Option<&mut AgentThread>,
    ) -> Result<AgentRunResponse> {
        let input = messages
            .iter()
            .map(ChatMessage::text)
            .collect::<Vec<_>>()
            .join(" ");
        let reply = format!("{}{}", self.prefix, input.trim());
        Ok(AgentRunResponse {
            messages: vec![ChatMessage::assistant(reply)],
            usage_details: self.usage.clone(),
            ..Default::default()
        })
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

/// A single-executor workflow that yields `"workflow: {input}"` as its output.
pub fn echo_workflow() -> Workflow {
    let echo = FunctionExecutor::new("echo", |msg: Value, ctx| async move {
        let text = msg.as_str().unwrap_or_default().to_string();
        ctx.yield_output(Value::String(format!("workflow: {text}")))
            .await?;
        Ok(())
    });
    WorkflowBuilder::new()
        .add_executor(Arc::new(echo))
        .set_start("echo")
        .name("Echo Workflow")
        .description("Echoes its input")
        .build()
        .expect("workflow builds")
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

/// Run one request against `app` and return `(status, body_bytes)`.
pub async fn send(app: Router, request: Request<Body>) -> (StatusCode, Vec<u8>) {
    let response = app.oneshot(request).await.expect("router responds");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body collects");
    (status, bytes.to_vec())
}

/// `GET uri`, parsing the JSON body.
pub async fn get_json(app: Router, uri: &str) -> (StatusCode, Value) {
    let request = Request::builder()
        .uri(uri)
        .body(Body::empty())
        .expect("request builds");
    let (status, bytes) = send(app, request).await;
    (status, parse_json(&bytes))
}

/// `POST uri` with a JSON body, parsing the JSON response.
pub async fn post_json(app: Router, uri: &str, body: &Value) -> (StatusCode, Value) {
    let request = Request::builder()
        .method("POST")
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(body).unwrap()))
        .expect("request builds");
    let (status, bytes) = send(app, request).await;
    (status, parse_json(&bytes))
}

/// `POST uri` with a raw string body, returning the raw text response (for SSE
/// and malformed-payload tests).
pub async fn post_raw(app: Router, uri: &str, body: String) -> (StatusCode, String) {
    let request = Request::builder()
        .method("POST")
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .expect("request builds");
    let (status, bytes) = send(app, request).await;
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

fn parse_json(bytes: &[u8]) -> Value {
    serde_json::from_slice(bytes)
        .unwrap_or_else(|e| panic!("invalid JSON: {e}: {}", String::from_utf8_lossy(bytes)))
}

/// Extract the `data:` payloads from an SSE body, in order.
pub fn parse_sse(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(str::to_string)
        .collect()
}

/// Parse the SSE `data:` payloads, dropping the terminal `[DONE]`, into JSON.
pub fn parse_sse_json(text: &str) -> Vec<Value> {
    parse_sse(text)
        .into_iter()
        .filter(|d| d != "[DONE]")
        .map(|d| serde_json::from_str(&d).expect("SSE data is JSON"))
        .collect()
}
