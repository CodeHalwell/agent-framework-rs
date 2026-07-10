//! The chat client trait and the automatic function-invocation loop.
//!
//! Rust equivalent of `agent_framework._clients` plus the tool loop from
//! `_tools.use_function_invocation`.

use async_trait::async_trait;
use futures::stream::{self, Stream, StreamExt};
use serde_json::Value;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use tracing::Instrument;

use crate::error::{Error, Result};
use crate::tools::{FunctionInvocationConfig, ToolDefinition};
use crate::types::{
    ChatMessage, ChatOptions, ChatResponse, ChatResponseUpdate, Content,
    FunctionApprovalRequestContent, FunctionApprovalResponseContent, FunctionCallContent,
    FunctionResultContent, Role, ToolMode,
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

/// The exact rejection payload Python emits for a denied tool call.
const REJECTION_MESSAGE: &str = "Error: Tool call invocation was rejected by user.";

/// Execute a single requested tool call, wrapped in an `execute_tool` span.
///
/// Returns `(is_error, result)`. `terminate_on_unknown` turns an unknown-tool
/// call into a hard error (propagated) rather than an error result.
async fn execute_tool_call(
    tool: Option<ToolDefinition>,
    call: &FunctionCallContent,
    include_detailed_errors: bool,
    terminate_on_unknown: bool,
) -> Result<(bool, FunctionResultContent)> {
    match tool {
        None => {
            if terminate_on_unknown {
                return Err(Error::tool(format!("unknown tool: {}", call.name)));
            }
            Ok((
                true,
                FunctionResultContent {
                    call_id: call.call_id.clone(),
                    result: None,
                    exception: Some(format!("tool '{}' not found", call.name)),
                },
            ))
        }
        Some(def) => {
            // Reject unparseable arguments rather than silently invoking the tool
            // with null/default input.
            let args = match call.parse_arguments() {
                Ok(m) => Value::Object(m.into_iter().collect()),
                Err(e) => {
                    let msg = if include_detailed_errors {
                        format!("invalid tool arguments: {e}")
                    } else {
                        "invalid tool arguments".to_string()
                    };
                    return Ok((
                        true,
                        FunctionResultContent {
                            call_id: call.call_id.clone(),
                            result: None,
                            exception: Some(msg),
                        },
                    ));
                }
            };
            let exec = def.executor.as_ref().unwrap().clone();
            let span = crate::observability::tool_span(&def.name, &call.call_id);
            let outcome = async move {
                let result = exec.invoke(args).await;
                if let Err(e) = &result {
                    tracing::Span::current().record(
                        crate::observability::attr::ERROR_TYPE,
                        crate::observability::error_type(e).as_str(),
                    );
                }
                result
            }
            .instrument(span)
            .await;
            match outcome {
                Ok(result) => Ok((
                    false,
                    FunctionResultContent {
                        call_id: call.call_id.clone(),
                        result: Some(result),
                        exception: None,
                    },
                )),
                Err(e) => {
                    let msg = if include_detailed_errors {
                        format!("{e}")
                    } else {
                        "tool execution failed".to_string()
                    };
                    Ok((
                        true,
                        FunctionResultContent {
                            call_id: call.call_id.clone(),
                            result: None,
                            exception: Some(msg),
                        },
                    ))
                }
            }
        }
    }
}

/// Collect all function-approval responses present in a conversation.
fn collect_approval_responses(messages: &[ChatMessage]) -> Vec<FunctionApprovalResponseContent> {
    let mut out = Vec::new();
    for msg in messages {
        for content in &msg.contents {
            if let Content::FunctionApprovalResponse(resp) = content {
                out.push(resp.clone());
            }
        }
    }
    out
}

/// Rewrite approval request/response contents in place, mirroring Python's
/// `_replace_approval_contents_with_results`.
///
/// * A [`FunctionApprovalRequestContent`] becomes its embedded
///   [`FunctionCallContent`], unless that call already exists in the same
///   message (a duplicate), in which case the request is removed.
/// * An approved [`FunctionApprovalResponseContent`] becomes the corresponding
///   result (correlated strictly by call id) and the message role becomes
///   `tool`.
/// * A rejected response becomes a [`FunctionResultContent`] carrying the
///   rejection payload, and the message role becomes `tool`.
fn replace_approval_contents_with_results(
    messages: &mut [ChatMessage],
    approved_results: &HashMap<String, FunctionResultContent>,
) {
    for msg in messages.iter_mut() {
        let existing_call_ids: std::collections::HashSet<String> = msg
            .contents
            .iter()
            .filter_map(Content::as_function_call)
            .filter(|fc| !fc.call_id.is_empty())
            .map(|fc| fc.call_id.clone())
            .collect();

        let mut to_remove: Vec<usize> = Vec::new();
        let mut set_role_tool = false;

        for (idx, content) in msg.contents.iter_mut().enumerate() {
            match content {
                Content::FunctionApprovalRequest(req) => {
                    if existing_call_ids.contains(&req.function_call.call_id) {
                        to_remove.push(idx);
                    } else {
                        *content = Content::FunctionCall(req.function_call.clone());
                    }
                }
                Content::FunctionApprovalResponse(resp) => {
                    let call_id = resp.function_call.call_id.clone();
                    if resp.approved {
                        if let Some(result) = approved_results.get(&call_id) {
                            *content = Content::FunctionResult(result.clone());
                            set_role_tool = true;
                        }
                    } else {
                        *content = Content::FunctionResult(FunctionResultContent {
                            call_id,
                            result: Some(Value::String(REJECTION_MESSAGE.to_string())),
                            exception: None,
                        });
                        set_role_tool = true;
                    }
                }
                _ => {}
            }
        }

        for idx in to_remove.into_iter().rev() {
            msg.contents.remove(idx);
        }
        if set_role_tool {
            msg.role = Role::tool();
        }
    }
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
            // Process any function-approval responses supplied in the input:
            // execute the approved calls and splice their results into the
            // conversation (mirrors Python's `_collect_approval_responses` +
            // `_replace_approval_contents_with_results`).
            let approval_responses = collect_approval_responses(&conversation);
            if !approval_responses.is_empty() {
                let mut approved_results: HashMap<String, FunctionResultContent> = HashMap::new();
                let mut had_error = false;
                for resp in &approval_responses {
                    if !resp.approved {
                        continue;
                    }
                    let call = &resp.function_call;
                    let tool = tools.iter().find(|t| t.name == call.name).cloned();
                    let (is_error, content) = execute_tool_call(
                        tool,
                        call,
                        self.config.include_detailed_errors,
                        self.config.terminate_on_unknown_calls,
                    )
                    .await?;
                    had_error |= is_error;
                    approved_results.insert(content.call_id.clone(), content);
                }
                replace_approval_contents_with_results(&mut conversation, &approved_results);
                if had_error {
                    consecutive_errors += 1;
                    if consecutive_errors > self.config.max_consecutive_errors_per_request {
                        options.tool_choice = Some(ToolMode::None);
                    }
                }
            }

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

            // Human-in-the-loop gate: if *any* requested tool requires approval,
            // defer *all* calls (matching Python) and return an assistant message
            // that carries the original calls plus one approval request each.
            let needs_approval = calls.iter().any(|c| {
                tools
                    .iter()
                    .find(|t| t.name == c.name)
                    .map(ToolDefinition::requires_approval)
                    .unwrap_or(false)
            });
            if needs_approval {
                let mut resp = response;
                let approval_contents: Vec<Content> = calls
                    .iter()
                    .map(|c| {
                        Content::FunctionApprovalRequest(FunctionApprovalRequestContent {
                            id: c.call_id.clone(),
                            function_call: c.clone(),
                        })
                    })
                    .collect();
                if let Some(m) = resp
                    .messages
                    .iter_mut()
                    .rev()
                    .find(|m| m.role == Role::assistant())
                {
                    m.contents.extend(approval_contents);
                } else {
                    resp.messages.push(ChatMessage::with_contents(
                        Role::assistant(),
                        approval_contents,
                    ));
                }
                let mut msgs = std::mem::take(&mut carried);
                msgs.append(&mut resp.messages);
                resp.messages = msgs;
                return Ok(resp);
            }

            // Record the assistant message(s) that requested the calls.
            carried.extend(response.messages.iter().cloned());

            // Execute all calls concurrently: the model may emit several
            // parallel tool calls, and I/O-bound tools should not be serialized.
            let invocations = calls.iter().map(|call| {
                let tool = tools.iter().find(|t| t.name == call.name).cloned();
                let call = call.clone();
                let include_detailed_errors = self.config.include_detailed_errors;
                let terminate_on_unknown = self.config.terminate_on_unknown_calls;
                async move {
                    execute_tool_call(tool, &call, include_detailed_errors, terminate_on_unknown)
                        .await
                }
            });

            let outcomes = futures::future::try_join_all(invocations).await?;
            let mut result_contents: Vec<Content> = Vec::with_capacity(outcomes.len());
            let mut had_error = false;
            for (is_error, content) in outcomes {
                had_error |= is_error;
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

            let tool_message = ChatMessage::with_contents(Role::tool(), result_contents);
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
        // Each message is replayed as its own update with a stable, distinct
        // `message_id` so that consumers re-aggregating via
        // `ChatResponse::from_updates` keep the messages separate rather than
        // merging the tool-call and final assistant messages by role.
        let response = self.get_response(messages, options).await?;
        let updates: Vec<Result<ChatResponseUpdate>> = response
            .messages
            .into_iter()
            .enumerate()
            .map(|(i, m)| {
                let message_id = m.message_id.clone().or_else(|| Some(format!("replay-{i}")));
                Ok(ChatResponseUpdate {
                    contents: m.contents,
                    role: Some(m.role),
                    author_name: m.author_name,
                    message_id,
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
