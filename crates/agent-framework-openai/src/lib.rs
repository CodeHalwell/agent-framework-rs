//! # agent-framework-openai
//!
//! An OpenAI (and OpenAI-compatible) [`ChatClient`] for `agent-framework-rs`.
//!
//! Works against the OpenAI Chat Completions API and any compatible endpoint
//! (Azure OpenAI, Ollama, together.ai, local servers, …) by overriding the
//! base URL.
//!
//! ```no_run
//! use agent_framework_openai::OpenAIClient;
//! use agent_framework_core::prelude::*;
//!
//! # async fn demo() -> Result<()> {
//! let client = OpenAIClient::new("sk-...", "gpt-4o-mini");
//! let agent = ChatAgent::builder(client)
//!     .instructions("You are concise.")
//!     .build();
//! let reply = agent.run_once("Say hi").await?;
//! println!("{}", reply.text());
//! # Ok(())
//! # }
//! ```

mod convert;

use std::sync::Arc;

use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::error::{Error, Result};
use agent_framework_core::types::{
    ChatMessage, ChatOptions, ChatResponse, ChatResponseUpdate, Content, FinishReason,
    FunctionArguments, FunctionCallContent, Role, TextContent,
};
use futures::StreamExt;
use serde_json::{json, Map, Value};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// An OpenAI (or OpenAI-compatible) chat client.
#[derive(Clone)]
pub struct OpenAIClient {
    inner: Arc<Inner>,
}

#[derive(Clone)]
struct Inner {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
    organization: Option<String>,
}

impl OpenAIClient {
    /// Create a client for the given API key and default model.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(Inner {
                http: reqwest::Client::new(),
                api_key: api_key.into(),
                base_url: DEFAULT_BASE_URL.to_string(),
                model: model.into(),
                organization: None,
            }),
        }
    }

    /// Build a client from the `OPENAI_API_KEY` (and optional
    /// `OPENAI_BASE_URL`) environment variables.
    pub fn from_env(model: impl Into<String>) -> Result<Self> {
        let key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| Error::Configuration("OPENAI_API_KEY is not set".into()))?;
        let mut client = Self::new(key, model);
        if let Ok(base) = std::env::var("OPENAI_BASE_URL") {
            client = client.with_base_url(base);
        }
        Ok(client)
    }

    /// Override the base URL (for Azure OpenAI or compatible servers).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).base_url = base_url.into();
        self
    }

    /// Set the organization header.
    pub fn with_organization(mut self, org: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).organization = Some(org.into());
        self
    }

    fn build_body(&self, messages: &[ChatMessage], options: &ChatOptions, stream: bool) -> Value {
        let mut body = Map::new();
        let model = options
            .model_id
            .clone()
            .unwrap_or_else(|| self.inner.model.clone());
        body.insert("model".into(), json!(model));
        body.insert(
            "messages".into(),
            json!(convert::messages_to_openai(messages)),
        );
        convert::apply_options(&mut body, options);
        let (tools, tool_choice) = convert::tools_to_openai(options);
        if let Some(tools) = tools {
            body.insert("tools".into(), tools);
        }
        if let Some(choice) = tool_choice {
            body.insert("tool_choice".into(), choice);
        }
        if stream {
            body.insert("stream".into(), json!(true));
            body.insert("stream_options".into(), json!({ "include_usage": true }));
        }
        Value::Object(body)
    }

    async fn post(&self, body: &Value) -> Result<reqwest::Response> {
        let url = format!(
            "{}/chat/completions",
            self.inner.base_url.trim_end_matches('/')
        );
        let mut req = self
            .inner
            .http
            .post(&url)
            .bearer_auth(&self.inner.api_key)
            .json(body);
        if let Some(org) = &self.inner.organization {
            req = req.header("OpenAI-Organization", org);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| Error::service(format!("request failed: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(Error::service(format!("OpenAI API error {status}: {text}")));
        }
        Ok(resp)
    }

    /// The default model id.
    pub fn model(&self) -> &str {
        &self.inner.model
    }
}

#[async_trait::async_trait]
impl ChatClient for OpenAIClient {
    async fn get_response(
        &self,
        messages: Vec<ChatMessage>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        let body = self.build_body(&messages, &options, false);
        let resp = self.post(&body).await?;
        let value: Value = resp
            .json()
            .await
            .map_err(|e| Error::service(format!("invalid response json: {e}")))?;
        Ok(convert::parse_response(&value))
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<ChatMessage>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let body = self.build_body(&messages, &options, true);
        let resp = self.post(&body).await?;
        Ok(parse_sse_stream(resp).boxed())
    }

    fn model_id(&self) -> Option<&str> {
        Some(&self.inner.model)
    }
}

type ByteStream =
    std::pin::Pin<Box<dyn futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Send>>;

/// Turn an SSE HTTP response into a stream of [`ChatResponseUpdate`]s.
fn parse_sse_stream(
    resp: reqwest::Response,
) -> impl futures::Stream<Item = Result<ChatResponseUpdate>> + Send {
    let byte_stream: ByteStream = Box::pin(resp.bytes_stream());
    futures::stream::unfold(
        (
            byte_stream,
            String::new(),
            Vec::<ChatResponseUpdate>::new(),
            false,
        ),
        |(mut byte_stream, mut buffer, mut queued, done)| async move {
            loop {
                if !queued.is_empty() {
                    let update = queued.remove(0);
                    return Some((Ok(update), (byte_stream, buffer, queued, done)));
                }
                if done {
                    return None;
                }
                match byte_stream.next().await {
                    Some(Ok(bytes)) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));
                        while let Some(pos) = buffer.find('\n') {
                            let line = buffer[..pos].trim().to_string();
                            buffer.drain(..=pos);
                            if let Some(data) = line.strip_prefix("data:") {
                                let data = data.trim();
                                if data == "[DONE]" {
                                    return drain_or_end(byte_stream, buffer, queued);
                                }
                                if let Ok(value) = serde_json::from_str::<Value>(data) {
                                    if let Some(update) = parse_delta(&value) {
                                        queued.push(update);
                                    }
                                }
                            }
                        }
                    }
                    Some(Err(e)) => {
                        return Some((
                            Err(Error::service(format!("stream error: {e}"))),
                            (byte_stream, buffer, queued, true),
                        ));
                    }
                    None => return drain_or_end(byte_stream, buffer, queued),
                }
            }
        },
    )
}

type SseState = (ByteStream, String, Vec<ChatResponseUpdate>, bool);

fn drain_or_end(
    byte_stream: ByteStream,
    buffer: String,
    mut queued: Vec<ChatResponseUpdate>,
) -> Option<(Result<ChatResponseUpdate>, SseState)> {
    if queued.is_empty() {
        None
    } else {
        let first = queued.remove(0);
        Some((Ok(first), (byte_stream, buffer, queued, true)))
    }
}

/// Parse one streamed chunk (`chat.completion.chunk`) into an update.
fn parse_delta(value: &Value) -> Option<ChatResponseUpdate> {
    let mut update = ChatResponseUpdate {
        response_id: value.get("id").and_then(Value::as_str).map(String::from),
        model_id: value.get("model").and_then(Value::as_str).map(String::from),
        ..Default::default()
    };

    let mut contents: Vec<Content> = Vec::new();

    if let Some(choice) = value
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
    {
        if let Some(delta) = choice.get("delta") {
            if let Some(r) = delta.get("role").and_then(Value::as_str) {
                update.role = Some(Role::new(r));
            }
            if let Some(text) = delta.get("content").and_then(Value::as_str) {
                if !text.is_empty() {
                    contents.push(Content::Text(TextContent::new(text)));
                }
            }
            if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for call in calls {
                    let id = call
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    let func = call.get("function");
                    let name = func
                        .and_then(|f| f.get("name"))
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    let args = func
                        .and_then(|f| f.get("arguments"))
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    contents.push(Content::FunctionCall(FunctionCallContent::new(
                        id,
                        name,
                        Some(FunctionArguments::Raw(args)),
                    )));
                }
            }
        }
        if let Some(fr) = choice.get("finish_reason").and_then(Value::as_str) {
            update.finish_reason = Some(FinishReason::new(fr));
        }
    }

    if update.role.is_none() {
        update.role = Some(Role::assistant());
    }
    update.contents = contents;
    Some(update)
}
