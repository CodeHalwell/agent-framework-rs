//! The chat client trait and the automatic function-invocation loop.
//!
//! Rust equivalent of `agent_framework._clients` plus the tool loop from
//! `_tools.use_function_invocation`.

use async_trait::async_trait;
use futures::stream::{self, Stream, StreamExt};
use std::pin::Pin;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::tools::{FunctionInvocationConfig, ToolDefinition};
use crate::types::{
    ChatMessage, ChatOptions, ChatResponse, ChatResponseUpdate, Content, FunctionResultContent,
    Role, ToolMode,
};

/// A boxed stream of streaming chat updates.
pub type ChatStream = Pin<Box<dyn Stream<Item = Result<ChatResponseUpdate>> + Send>>;

/// The interface every chat client implements.
///
/// Implementors provide [`ChatClient::get_response`] and
/// [`ChatClient::get_streaming_response`]; the framework layers tool invocation
/// and middleware on top via [`FunctionInvokingChatClient`].
#[async_trait]
pub trait ChatClient: Send + Sync {
    /// Get a complete (non-streaming) response.
    async fn get_response(
        &self,
        messages: Vec<ChatMessage>,
        options: ChatOptions,
    ) -> Result<ChatResponse>;

    /// Get a streaming response as a sequence of updates.
    async fn get_streaming_response(
        &self,
        messages: Vec<ChatMessage>,
        options: ChatOptions,
    ) -> Result<ChatStream>;

    /// The default model id for this client, if any.
    fn model_id(&self) -> Option<&str> {
        None
    }
}

/// Blanket impl so `Arc<dyn ChatClient>` and wrappers are usable as clients.
#[async_trait]
impl<T: ChatClient + ?Sized> ChatClient for Arc<T> {
    async fn get_response(
        &self,
        messages: Vec<ChatMessage>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        (**self).get_response(messages, options).await
    }
    async fn get_streaming_response(
        &self,
        messages: Vec<ChatMessage>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        (**self).get_streaming_response(messages, options).await
    }
    fn model_id(&self) -> Option<&str> {
        (**self).model_id()
    }
}

/// Wraps a [`ChatClient`] to automatically execute local tool calls in a loop,
/// mirroring `use_function_invocation`.
pub struct FunctionInvokingChatClient<C: ChatClient> {
    inner: C,
    config: FunctionInvocationConfig,
}

impl<C: ChatClient> FunctionInvokingChatClient<C> {
    pub fn new(inner: C) -> Self {
        Self {
            inner,
            config: FunctionInvocationConfig::default(),
        }
    }

    /// Override the function-invocation configuration.
    pub fn with_config(mut self, config: FunctionInvocationConfig) -> Self {
        self.config = config;
        self
    }

    /// A reference to the wrapped client.
    pub fn inner(&self) -> &C {
        &self.inner
    }

    async fn inner_get_response(
        &self,
        messages: Vec<ChatMessage>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        self.inner.get_response(messages, options).await
    }
}

/// Extract the executable tools from the options into a name→tool map.
fn executable_tools(options: &ChatOptions) -> Vec<ToolDefinition> {
    options
        .tools
        .iter()
        .filter(|t| t.is_executable())
        .cloned()
        .collect()
}

#[async_trait]
impl<C: ChatClient> ChatClient for FunctionInvokingChatClient<C> {
    async fn get_response(
        &self,
        messages: Vec<ChatMessage>,
        mut options: ChatOptions,
    ) -> Result<ChatResponse> {
        self.config.validate()?;
        let tools = executable_tools(&options);

        // Default tool choice to auto when tools are present and unset.
        if !options.tools.is_empty() && options.tool_choice.is_none() {
            options.tool_choice = Some(ToolMode::Auto);
        }

        if tools.is_empty() || !self.config.enabled {
            return self.inner_get_response(messages, options).await;
        }

        let mut conversation = messages;
        let mut carried: Vec<ChatMessage> = Vec::new();
        let mut consecutive_errors = 0usize;

        for _ in 0..self.config.max_iterations {
            let response = self
                .inner_get_response(conversation.clone(), options.clone())
                .await?;

            let calls: Vec<_> = response
                .messages
                .iter()
                .flat_map(|m| m.contents.iter())
                .filter_map(Content::as_function_call)
                .cloned()
                .collect();

            if calls.is_empty() {
                // Prepend the accumulated tool-interaction messages so the final
                // assistant message stays last.
                let mut final_resp = response;
                let mut msgs = std::mem::take(&mut carried);
                msgs.append(&mut final_resp.messages);
                final_resp.messages = msgs;
                return Ok(final_resp);
            }

            // Record the assistant message(s) that requested the calls.
            carried.extend(response.messages.iter().cloned());

            // Execute each call and collect results.
            let mut result_contents: Vec<Content> = Vec::new();
            let mut had_error = false;
            for call in &calls {
                let tool = tools.iter().find(|t| t.name == call.name);
                let content = match tool {
                    None => {
                        if self.config.terminate_on_unknown_calls {
                            return Err(Error::tool(format!("unknown tool: {}", call.name)));
                        }
                        had_error = true;
                        FunctionResultContent {
                            call_id: call.call_id.clone(),
                            result: None,
                            exception: Some(format!("tool '{}' not found", call.name)),
                        }
                    }
                    Some(def) => {
                        let args = call
                            .parse_arguments()
                            .map(|m| serde_json::Value::Object(m.into_iter().collect()))
                            .unwrap_or(serde_json::Value::Null);
                        let exec = def.executor.as_ref().unwrap();
                        match exec.invoke(args).await {
                            Ok(result) => FunctionResultContent {
                                call_id: call.call_id.clone(),
                                result: Some(result),
                                exception: None,
                            },
                            Err(e) => {
                                had_error = true;
                                let msg = if self.config.include_detailed_errors {
                                    format!("{e}")
                                } else {
                                    "tool execution failed".to_string()
                                };
                                FunctionResultContent {
                                    call_id: call.call_id.clone(),
                                    result: None,
                                    exception: Some(msg),
                                }
                            }
                        }
                    }
                };
                result_contents.push(Content::FunctionResult(content));
            }

            if had_error {
                consecutive_errors += 1;
                if consecutive_errors > self.config.max_consecutive_errors_per_request {
                    // Give up on tools and let the model answer directly.
                    options.tool_choice = Some(ToolMode::None);
                }
            } else {
                consecutive_errors = 0;
            }

            let tool_message = ChatMessage::with_contents(Role::tool(), result_contents.clone());
            carried.push(tool_message.clone());
            conversation.extend(response.messages);
            conversation.push(tool_message);
        }

        // Failsafe: one final call with tools disabled.
        options.tool_choice = Some(ToolMode::None);
        let mut final_resp = self.inner_get_response(conversation, options).await?;
        let mut msgs = std::mem::take(&mut carried);
        msgs.append(&mut final_resp.messages);
        final_resp.messages = msgs;
        Ok(final_resp)
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<ChatMessage>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let tools = executable_tools(&options);
        if tools.is_empty() || !self.config.enabled {
            return self.inner.get_streaming_response(messages, options).await;
        }
        // With tools, run the full loop then stream the aggregated result.
        let response = self.get_response(messages, options).await?;
        let updates: Vec<Result<ChatResponseUpdate>> = response
            .messages
            .into_iter()
            .map(|m| {
                Ok(ChatResponseUpdate {
                    contents: m.contents,
                    role: Some(m.role),
                    author_name: m.author_name,
                    ..Default::default()
                })
            })
            .collect();
        Ok(stream::iter(updates).boxed())
    }

    fn model_id(&self) -> Option<&str> {
        self.inner.model_id()
    }
}
