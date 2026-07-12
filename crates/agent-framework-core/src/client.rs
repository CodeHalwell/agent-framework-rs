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
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::Instrument;

use crate::error::{Error, Result};
use crate::middleware::{FunctionInvocationContext, MiddlewarePipeline, Terminal};
use crate::tools::{FunctionInvocationConfig, ToolDefinition, ToolKind};
use crate::types::{
    ChatMessage, ChatOptions, ChatResponse, ChatResponseUpdate, Content,
    FunctionApprovalRequestContent, FunctionApprovalResponseContent, FunctionCallContent,
    FunctionResultContent, Role, ToolMode, UsageContent,
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
    /// Middleware run around every individual tool call (mirrors Python's
    /// function-middleware pipeline, driven here instead of by a
    /// `use_function_invocation` decorator).
    function_middleware: MiddlewarePipeline<FunctionInvocationContext>,
}

impl<C: ChatClient> FunctionInvokingChatClient<C> {
    pub fn new(inner: C) -> Self {
        Self {
            inner,
            config: FunctionInvocationConfig::default(),
            function_middleware: MiddlewarePipeline::default(),
        }
    }

    /// Override the function-invocation configuration.
    pub fn with_config(mut self, config: FunctionInvocationConfig) -> Self {
        self.config = config;
        self
    }

    /// Configure the function-invocation middleware pipeline run around every
    /// tool call: middleware may inspect/rewrite
    /// [`FunctionInvocationContext::arguments`], short-circuit execution by
    /// setting [`FunctionInvocationContext::result`] (and either not calling
    /// `next`, or setting `terminate = true`), or observe a propagated
    /// execution error by matching on the `Result` returned from their own
    /// `next.run(...)` call. Replaces any previously configured middleware.
    pub fn with_function_middleware(
        mut self,
        middleware: Vec<Arc<crate::middleware::FunctionMiddleware>>,
    ) -> Self {
        self.function_middleware = MiddlewarePipeline::new(middleware);
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

/// Whether `tool` is a *declaration-only* function tool: a known function with
/// no local executor. Mirrors Python's `AIFunction.declaration_only`. A call to
/// such a tool is returned to the caller unexecuted (the frontend-tool pattern
/// that makes AG-UI client-side tools work). Hosted tools (web search, MCP, …)
/// are deliberately excluded — they are not function tools and a call whose
/// name matches none of the local function tools is treated as unknown, not
/// declaration-only, exactly as Python's `_get_tool_map` omits them.
fn is_declaration_only(tool: &ToolDefinition) -> bool {
    tool.kind == ToolKind::Function && tool.executor.is_none()
}

/// The exact rejection payload Python emits for a denied tool call.
const REJECTION_MESSAGE: &str = "Error: Tool call invocation was rejected by user.";

/// Execute a single requested tool call through the function-middleware
/// pipeline, with the actual invocation (wrapped in an `execute_tool` span)
/// as the pipeline's terminal handler.
///
/// Returns `(is_error, result)`. `terminate_on_unknown` turns an unknown-tool
/// call into a hard error (propagated) rather than an error result. Unknown
/// tools and unparseable arguments are rejected before middleware ever sees
/// them (there is no function to hand the pipeline in that case); once a
/// [`FunctionInvocationContext`] is built, middleware can rewrite
/// `arguments`, short-circuit by setting `result` (without calling `next`, or
/// with `terminate = true`), or observe an execution error by matching on the
/// `Result` their own `next.run(...)` call returns. A propagated error is
/// converted to the same `(true, FunctionResultContent { exception: .. })`
/// shape the direct-error path used before middleware existed, so
/// `include_detailed_errors` behaves identically either way.
async fn execute_tool_call(
    tool: Option<ToolDefinition>,
    call: &FunctionCallContent,
    include_detailed_errors: bool,
    terminate_on_unknown: bool,
    function_middleware: &MiddlewarePipeline<FunctionInvocationContext>,
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
            let tool_name = def.name.clone();
            let description = def.description.clone();
            let call_id = call.call_id.clone();
            let terminal: Terminal<FunctionInvocationContext> = Box::new(move |mut ctx| {
                Box::pin(async move {
                    if ctx.terminate {
                        return Ok(ctx);
                    }
                    let span = crate::observability::tool_span_ex(
                        &tool_name,
                        &call_id,
                        Some(&description),
                    );
                    let capture =
                        crate::observability::ObservabilityConfig::from_env().enable_sensitive_data;
                    crate::observability::record_tool_arguments(&span, &ctx.arguments, capture);
                    #[cfg(feature = "otel-metrics")]
                    let started = std::time::Instant::now();
                    let outcome = async {
                        let result = exec.invoke(ctx.arguments.clone()).await;
                        if let Err(e) = &result {
                            crate::observability::record_error(&tracing::Span::current(), e);
                        }
                        result
                    }
                    .instrument(span.clone())
                    .await;
                    #[cfg(feature = "otel-metrics")]
                    crate::observability::metrics::record_function_invocation_duration(
                        &tool_name,
                        started.elapsed(),
                        outcome
                            .as_ref()
                            .err()
                            .map(crate::observability::error_type)
                            .as_deref(),
                    );
                    if let Ok(value) = &outcome {
                        crate::observability::record_tool_result(&span, value, capture);
                    }
                    ctx.result = Some(outcome?);
                    Ok(ctx)
                }) as crate::tools::BoxFuture<Result<FunctionInvocationContext>>
            });

            let ctx = FunctionInvocationContext::new(call.name.clone(), args);
            match function_middleware.execute(ctx, terminal).await {
                Ok(ctx) => Ok((
                    false,
                    FunctionResultContent {
                        call_id: call.call_id.clone(),
                        result: Some(ctx.result.unwrap_or(Value::Null)),
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
        // After the tool loop settles, auto-populate `ChatResponse.value` from
        // the final text when a structured `response_format` was requested
        // (mirrors Python `try_parse_value`). This is the central non-streaming
        // fill point: it covers a bare `FunctionInvokingChatClient` and every
        // `ChatAgent` run (whose client is always wrapped in one). The tool
        // loop is run inside an `async move` block so its interior `return`s
        // funnel through this single fill/return path.
        let response_format = options.response_format.clone();
        let mut response: ChatResponse = async move {
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
                    let mut approved_results: HashMap<String, FunctionResultContent> =
                        HashMap::new();
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
                            &self.function_middleware,
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

                // A call whose result is already present in the same response
                // was executed by the provider (e.g. Anthropic server-side
                // web-search/code-execution/MCP `server_tool_use` blocks,
                // which arrive paired with their `*_tool_result`). Executing
                // it locally would produce a bogus "tool not found" — only
                // unresolved calls enter the local tool loop.
                let resolved_call_ids: std::collections::HashSet<&str> = response
                    .messages
                    .iter()
                    .flat_map(|m| m.contents.iter())
                    .filter_map(Content::as_function_result)
                    .map(|fr| fr.call_id.as_str())
                    .collect();
                let calls: Vec<_> = response
                    .messages
                    .iter()
                    .flat_map(|m| m.contents.iter())
                    .filter_map(Content::as_function_call)
                    .filter(|fc| !resolved_call_ids.contains(fc.call_id.as_str()))
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

                // Declaration-only calls: a call targeting a KNOWN tool that has
                // no local executor (declaration-only — e.g. an AG-UI frontend
                // tool, or a per-run `additional_tools` entry) terminates the
                // loop and returns the response with the `FunctionCallContent`
                // intact, so the caller can execute it. Mirrors Python's
                // `_try_execute_function_calls` `declaration_only` branch
                // (`_tools.py:1396-1420`): if *any* requested call is
                // declaration-only, the whole response is returned unexecuted.
                // A genuinely unknown tool name is NOT declaration-only and
                // keeps today's not-found handling in `execute_tool_call`.
                let has_declaration_only = calls.iter().any(|c| {
                    options
                        .tools
                        .iter()
                        .any(|t| t.name == c.name && is_declaration_only(t))
                });
                if has_declaration_only {
                    let mut resp = response;
                    let mut msgs = std::mem::take(&mut carried);
                    msgs.append(&mut resp.messages);
                    resp.messages = msgs;
                    return Ok(resp);
                }

                // Record the assistant message(s) that requested the calls.
                carried.extend(response.messages.iter().cloned());
                let response_conversation_id = response.conversation_id.clone();

                // Execute all calls concurrently: the model may emit several
                // parallel tool calls, and I/O-bound tools should not be serialized.
                let invocations = calls.iter().map(|call| {
                    let tool = tools.iter().find(|t| t.name == call.name).cloned();
                    let call = call.clone();
                    let include_detailed_errors = self.config.include_detailed_errors;
                    let terminate_on_unknown = self.config.terminate_on_unknown_calls;
                    let function_middleware = self.function_middleware.clone();
                    async move {
                        execute_tool_call(
                            tool,
                            &call,
                            include_detailed_errors,
                            terminate_on_unknown,
                            &function_middleware,
                        )
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
                match response_conversation_id {
                    // A service-managed client that created (or continued) the
                    // conversation now holds the history server-side. Propagate
                    // its id so the follow-up tool-output submission targets the
                    // right thread — without this, Assistants / Azure AI reject
                    // the submission because `conversation_id` is still `None` —
                    // and send ONLY the new tool results next turn rather than
                    // re-sending the whole history (mirrors Python
                    // `_tools.py:1635-1637, 1695-1699`).
                    Some(cid) => {
                        options.conversation_id = Some(cid);
                        conversation = vec![tool_message];
                    }
                    // Stateless client (e.g. Chat Completions): accumulate and
                    // re-send the full history each turn.
                    None => {
                        conversation.extend(response.messages);
                        conversation.push(tool_message);
                    }
                }
            }

            // Failsafe: one final call with tools disabled.
            options.tool_choice = Some(ToolMode::None);
            let mut final_resp = self.inner_get_response(conversation, options).await?;
            let mut msgs = std::mem::take(&mut carried);
            msgs.append(&mut final_resp.messages);
            final_resp.messages = msgs;
            Ok(final_resp)
        }
        .await?;
        response.try_parse_value(response_format.as_ref());
        Ok(response)
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
        // Response-level metadata must survive the replay so re-aggregation
        // (and the agent's thread adoption) sees it: ids on every update,
        // and usage/finish-reason on the final one (usage rides as a
        // `Content::Usage` item, which `absorb_update` folds into
        // `usage_details` rather than the message contents — the same shape
        // providers use for their terminal stream chunk).
        let conversation_id = response.conversation_id.clone();
        let response_id = response.response_id.clone();
        let finish_reason = response.finish_reason.clone();
        let usage_details = response.usage_details.clone();
        let last = response.messages.len().saturating_sub(1);
        let mut updates: Vec<Result<ChatResponseUpdate>> = response
            .messages
            .into_iter()
            .enumerate()
            .map(|(i, m)| {
                let message_id = m.message_id.clone().or_else(|| Some(format!("replay-{i}")));
                let mut contents = m.contents;
                let is_last = i == last;
                if is_last {
                    if let Some(usage) = usage_details.clone() {
                        contents.push(Content::Usage(UsageContent { details: usage }));
                    }
                }
                Ok(ChatResponseUpdate {
                    contents,
                    role: Some(m.role),
                    author_name: m.author_name,
                    message_id,
                    conversation_id: conversation_id.clone(),
                    response_id: response_id.clone(),
                    finish_reason: is_last.then(|| finish_reason.clone()).flatten(),
                    ..Default::default()
                })
            })
            .collect();
        // A messageless response (unusual, but possible) still carries its
        // terminal metadata in one trailing update.
        if updates.is_empty() && (usage_details.is_some() || finish_reason.is_some()) {
            let contents = usage_details
                .map(|u| vec![Content::Usage(UsageContent { details: u })])
                .unwrap_or_default();
            updates.push(Ok(ChatResponseUpdate {
                contents,
                role: Some(Role::assistant()),
                conversation_id,
                response_id,
                finish_reason,
                ..Default::default()
            }));
        }
        Ok(stream::iter(updates).boxed())
    }

    fn model_id(&self) -> Option<&str> {
        self.inner.model_id()
    }
}

// ---------------------------------------------------------------------------
// Retry / backoff layer
// ---------------------------------------------------------------------------

/// Which errors a [`RetryPolicy`] considers retryable.
#[derive(Clone)]
pub enum RetryOn {
    /// The built-in default predicate (see [`RetryPolicy`] docs for the exact
    /// rule): retries HTTP `408`/`429`/`5xx` ([`Error::ServiceStatus`]) and
    /// transport-ish [`Error::Service`] failures (timeouts / connection
    /// errors). Never retries [`Error::ServiceInvalidAuth`],
    /// [`Error::ServiceInvalidRequest`], or [`Error::ServiceContentFilter`] —
    /// authentication/authorization failures, malformed requests, and
    /// content-filter refusals are non-transient, so retrying would just
    /// repeat the same rejection.
    Default,
    /// A fully custom predicate deciding, per error, whether to retry.
    Predicate(Arc<dyn Fn(&Error) -> bool + Send + Sync>),
}

impl std::fmt::Debug for RetryOn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RetryOn::Default => f.write_str("RetryOn::Default"),
            RetryOn::Predicate(_) => f.write_str("RetryOn::Predicate(..)"),
        }
    }
}

impl RetryOn {
    /// A custom retry predicate.
    pub fn predicate<F>(f: F) -> Self
    where
        F: Fn(&Error) -> bool + Send + Sync + 'static,
    {
        RetryOn::Predicate(Arc::new(f))
    }

    fn should_retry(&self, err: &Error) -> bool {
        match self {
            RetryOn::Default => default_should_retry(err),
            RetryOn::Predicate(p) => p(err),
        }
    }
}

/// The default retryability rule used by [`RetryOn::Default`].
///
/// Retries when either:
/// * the error is an [`Error::ServiceStatus`] whose status is `408`
///   (Request Timeout), `429` (Too Many Requests), or any `5xx`; or
/// * the error is an [`Error::Service`] whose (lowercased) message contains one
///   of the transport-failure markers the provider clients emit — `"request
///   failed"` (the prefix wrapping every `reqwest` send error: DNS, connect,
///   timeout, reset), `"timed out"`, `"timeout"`, `"connection"`, or `"stream
///   error"`.
///
/// Everything else (4xx other than 408/429, parse errors, tool/workflow errors,
/// non-transport service errors) is treated as non-retryable. This explicitly
/// includes [`Error::ServiceInvalidAuth`], [`Error::ServiceInvalidRequest`],
/// and [`Error::ServiceContentFilter`] — authentication/authorization
/// failures, malformed requests, and content-filter refusals are
/// non-transient, so retrying would just repeat the same rejection. None of
/// the three carry a status via [`Error::status`], so they fall through to
/// the final `_ => false` below (there's no dedicated match arm for them:
/// merging one in would just duplicate that `false`, which `clippy` flags as
/// `match_same_arms`).
fn default_should_retry(err: &Error) -> bool {
    if let Some(status) = err.status() {
        return status == 408 || status == 429 || (500..600).contains(&status);
    }
    match err {
        Error::Service(msg) => {
            let m = msg.to_lowercase();
            m.contains("request failed")
                || m.contains("timed out")
                || m.contains("timeout")
                || m.contains("connection")
                || m.contains("stream error")
        }
        _ => false,
    }
}

/// Policy controlling [`RetryingChatClient`] backoff.
///
/// Delays grow exponentially from [`initial_delay`](Self::initial_delay) by
/// [`backoff_multiplier`](Self::backoff_multiplier) per attempt, are capped at
/// [`max_delay`](Self::max_delay), and are then reduced by up to
/// [`jitter`](Self::jitter) (a fraction of the delay). When the failing error
/// carries a server `Retry-After` (see [`Error::retry_after`]) that value is
/// used instead of the computed backoff (still capped by `max_delay`, and not
/// jittered — it is an explicit server instruction).
#[derive(Clone, Debug)]
pub struct RetryPolicy {
    /// Maximum number of *retries* after the initial attempt (default `3`, so
    /// up to four total attempts).
    pub max_retries: usize,
    /// Base delay before the first retry (default `500ms`).
    pub initial_delay: Duration,
    /// Upper bound on any single delay, also capping a server `Retry-After`
    /// (default `30s`).
    pub max_delay: Duration,
    /// Exponential growth factor applied per retry (default `2.0`).
    pub backoff_multiplier: f64,
    /// Jitter as a fraction in `0.0..=1.0` (default `0.3`): the computed delay
    /// is multiplied by `1 - jitter * r` for a per-attempt random `r` in
    /// `[0, 1)`. `0.0` disables jitter (fully deterministic delays).
    pub jitter: f64,
    /// Which errors to retry (default [`RetryOn::Default`]).
    pub retry_on: RetryOn,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter: 0.3,
            retry_on: RetryOn::Default,
        }
    }
}

impl RetryPolicy {
    /// A policy with the given retry count and otherwise-default backoff.
    pub fn with_max_retries(max_retries: usize) -> Self {
        Self {
            max_retries,
            ..Self::default()
        }
    }

    /// Set the base delay before the first retry.
    pub fn initial_delay(mut self, delay: Duration) -> Self {
        self.initial_delay = delay;
        self
    }

    /// Set the per-delay cap (also caps a server `Retry-After`).
    pub fn max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    /// Set the exponential growth factor.
    pub fn backoff_multiplier(mut self, multiplier: f64) -> Self {
        self.backoff_multiplier = multiplier;
        self
    }

    /// Set the jitter fraction (clamped to `0.0..=1.0`).
    pub fn jitter(mut self, jitter: f64) -> Self {
        self.jitter = jitter.clamp(0.0, 1.0);
        self
    }

    /// Set the retryability rule.
    pub fn retry_on(mut self, retry_on: RetryOn) -> Self {
        self.retry_on = retry_on;
        self
    }

    /// The delay to wait before a retry, given the 1-based `attempt` number
    /// (attempt `1` is the first retry) and the error that triggered it.
    fn delay_for(&self, attempt: usize, err: &Error) -> Duration {
        // A server-advised `Retry-After` wins over computed backoff (capped by
        // `max_delay`, not jittered — it is an explicit instruction).
        if let Some(secs) = err.retry_after() {
            let capped = secs.min(self.max_delay.as_secs_f64()).max(0.0);
            return Duration::from_secs_f64(capped);
        }
        let exp = self.backoff_multiplier.powi((attempt - 1) as i32);
        let base = self.initial_delay.as_secs_f64() * exp;
        let capped = base.min(self.max_delay.as_secs_f64());
        let jittered = capped * jitter_factor(self.jitter);
        Duration::from_secs_f64(jittered.max(0.0))
    }
}

/// A cheap jitter multiplier in `[1 - jitter, 1.0]`, without a `rand`
/// dependency: entropy comes from the current wall-clock nanoseconds mixed
/// with a process-lifetime counter (so repeated calls within the same
/// nanosecond still differ). `jitter <= 0` returns `1.0` (no jitter).
fn jitter_factor(jitter: f64) -> f64 {
    let jitter = jitter.clamp(0.0, 1.0);
    if jitter == 0.0 {
        return 1.0;
    }
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let mixed = nanos ^ COUNTER.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed);
    // Map to [0, 1) via the top 53 bits (f64 mantissa width).
    let r = (mixed >> 11) as f64 / ((1u64 << 53) as f64);
    1.0 - jitter * r
}

/// A [`ChatClient`] decorator that retries transient failures with exponential
/// backoff, honoring a server `Retry-After` when present.
///
/// Wraps any inner [`ChatClient`] and re-issues the request per its
/// [`RetryPolicy`]. For streaming, only the *initial connection* is retried:
/// if establishing the stream (or its very first item, before anything is
/// yielded to the consumer) fails with a retryable error, the connection is
/// re-attempted; once the first update flows, later stream errors propagate
/// unchanged.
///
/// ```no_run
/// # use std::time::Duration;
/// # use agent_framework_core::client::{RetryingChatClient, RetryPolicy};
/// # use agent_framework_core::prelude::*;
/// # fn demo(inner: impl ChatClient + 'static) {
/// let client = RetryingChatClient::new(inner)
///     .with_policy(RetryPolicy::with_max_retries(5).initial_delay(Duration::from_millis(200)));
/// # let _ = client;
/// # }
/// ```
pub struct RetryingChatClient<C: ChatClient> {
    inner: C,
    policy: RetryPolicy,
}

impl<C: ChatClient> RetryingChatClient<C> {
    /// Wrap `inner` with the default [`RetryPolicy`].
    pub fn new(inner: C) -> Self {
        Self {
            inner,
            policy: RetryPolicy::default(),
        }
    }

    /// Set the retry policy (builder-style).
    pub fn with_policy(mut self, policy: RetryPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// A reference to the wrapped client.
    pub fn inner(&self) -> &C {
        &self.inner
    }

    /// A reference to the active retry policy.
    pub fn policy(&self) -> &RetryPolicy {
        &self.policy
    }

    /// Sleep before a retry, emitting a tracing warning describing the attempt.
    async fn backoff(&self, attempt: usize, err: &Error) {
        let delay = self.policy.delay_for(attempt, err);
        tracing::warn!(
            attempt,
            max_retries = self.policy.max_retries,
            delay_ms = delay.as_millis() as u64,
            retry_after = err.retry_after(),
            status = err.status(),
            error = %err,
            "retrying chat request after transient error"
        );
        tokio::time::sleep(delay).await;
    }
}

#[async_trait]
impl<C: ChatClient> ChatClient for RetryingChatClient<C> {
    async fn get_response(
        &self,
        messages: Vec<ChatMessage>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        let mut attempt = 0usize;
        loop {
            match self
                .inner
                .get_response(messages.clone(), options.clone())
                .await
            {
                Ok(resp) => return Ok(resp),
                Err(e) => {
                    if attempt >= self.policy.max_retries || !self.policy.retry_on.should_retry(&e)
                    {
                        return Err(e);
                    }
                    attempt += 1;
                    self.backoff(attempt, &e).await;
                }
            }
        }
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<ChatMessage>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let mut attempt = 0usize;
        loop {
            let established = self
                .inner
                .get_streaming_response(messages.clone(), options.clone())
                .await;
            match established {
                // The stream opened: peek its first item. An error there (with
                // nothing yet yielded to the consumer) is still an
                // initial-connection failure and is eligible for retry; any Ok
                // item — or a non-retryable / retries-exhausted error — is
                // handed back with the rest of the stream chained after it.
                Ok(mut stream) => match stream.next().await {
                    Some(Err(e))
                        if attempt < self.policy.max_retries
                            && self.policy.retry_on.should_retry(&e) =>
                    {
                        attempt += 1;
                        self.backoff(attempt, &e).await;
                        continue;
                    }
                    Some(first) => {
                        let head = stream::once(async move { first });
                        return Ok(head.chain(stream).boxed());
                    }
                    None => return Ok(stream::empty().boxed()),
                },
                // The stream never opened (e.g. a non-success HTTP status).
                Err(e) => {
                    if attempt >= self.policy.max_retries || !self.policy.retry_on.should_retry(&e)
                    {
                        return Err(e);
                    }
                    attempt += 1;
                    self.backoff(attempt, &e).await;
                }
            }
        }
    }

    fn model_id(&self) -> Option<&str> {
        self.inner.model_id()
    }
}
