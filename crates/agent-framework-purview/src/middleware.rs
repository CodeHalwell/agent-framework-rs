//! [`PurviewAgentMiddleware`] and [`PurviewChatMiddleware`]: enforce Purview
//! policy on both the outgoing prompt and the model/agent response. Mirrors
//! Python's `PurviewPolicyMiddleware` (agent-level) and
//! `PurviewChatPolicyMiddleware` (chat-client-level) — see the crate docs
//! for which hook points these attach to and why both directions evaluate
//! with [`Activity::UploadText`](crate::models::Activity::UploadText).

use async_trait::async_trait;

use agent_framework_core::error::Result;
use agent_framework_core::middleware::{AgentContext, ChatContext, Middleware, Next};
use agent_framework_core::types::{AgentResponse, ChatResponse, Message, Role};

use crate::auth::TokenProvider;
use crate::client::PurviewClient;
use crate::processor::ContentProcessor;
use crate::settings::PurviewSettings;

/// Shared evaluation core for [`PurviewAgentMiddleware`] and
/// [`PurviewChatMiddleware`]: identical policy logic, only the surrounding
/// context type differs — matching how the two Python middleware classes
/// each build their own `PurviewClient`/`ScopedContentProcessor` but run the
/// exact same `process_messages` calls.
struct PurviewPolicyCore {
    processor: ContentProcessor,
    settings: PurviewSettings,
}

impl PurviewPolicyCore {
    fn new(token_provider: impl TokenProvider + 'static, settings: PurviewSettings) -> Self {
        let client = PurviewClient::new(token_provider, &settings);
        Self {
            processor: ContentProcessor::new(client),
            settings,
        }
    }

    /// Evaluate `messages`, folding a suppressed error into an "allow, carry
    /// the previously-known user id forward" outcome.
    ///
    /// Mirrors the two-tier `except PurviewPaymentRequiredError: ... except
    /// Exception: ...` structure in both Python middleware classes: a 402
    /// is checked against [`PurviewSettings::ignore_payment_required`]
    /// *independently of* [`PurviewSettings::ignore_exceptions`], which
    /// governs every other error. An unsuppressed error propagates (`?` at
    /// the call site), same as Python re-raising.
    async fn check(
        &self,
        messages: &[Message],
        provided_user_id: Option<&str>,
        phase: &'static str,
    ) -> Result<(bool, Option<String>)> {
        match self
            .processor
            .evaluate(messages, &self.settings, provided_user_id)
            .await
        {
            Ok(outcome) => Ok(outcome),
            Err(e) => {
                let suppress = if e.status() == Some(402) {
                    self.settings.ignore_payment_required
                } else {
                    self.settings.ignore_exceptions
                };
                if suppress {
                    tracing::error!(error = %e, phase, "Purview policy check failed; continuing (ignored per settings)");
                    Ok((false, provided_user_id.map(str::to_string)))
                } else {
                    Err(e)
                }
            }
        }
    }

    fn blocked_prompt_message(&self) -> Message {
        Message::new(Role::system(), self.settings.blocked_prompt_message.clone())
    }

    fn blocked_response_message(&self) -> Message {
        Message::new(
            Role::system(),
            self.settings.blocked_response_message.clone(),
        )
    }
}

/// Agent middleware enforcing Purview policy on both the outgoing prompt and
/// the agent's response. Mirrors Python's `PurviewPolicyMiddleware`.
///
/// - **Prompt (pre) check**: evaluates `ctx.messages`. If blocked,
///   short-circuits with a single `system`-role message
///   ([`PurviewSettings::blocked_prompt_message`]) and `ctx.terminate =
///   true` — the wrapped agent/chat-client is never called, matching this
///   crate's `ShortCircuitChat`-style test pattern in
///   `agent-framework-core`.
/// - **Response (post) check**: only runs when `!ctx.is_streaming` and a
///   result was produced (mirrors Python: "Streaming responses are not
///   supported for post-checks"). If blocked, `ctx.result` is *replaced*
///   with a single `system`-role message
///   ([`PurviewSettings::blocked_response_message`]) — unlike the prompt
///   check, `ctx.terminate` is not set here (there's nothing left to
///   terminate; the underlying call already happened), matching Python.
///
/// ```no_run
/// use agent_framework_core::prelude::*;
/// use agent_framework_purview::{PurviewAgentMiddleware, PurviewSettings, PurviewAppLocation, PurviewLocationType, StaticTokenProvider};
/// use std::sync::Arc;
///
/// # fn demo(client: impl ChatClient + 'static) {
/// let settings = PurviewSettings::new("My App")
///     .with_tenant_id("00000000-0000-0000-0000-000000000000")
///     .with_purview_app_location(PurviewAppLocation::new(
///         PurviewLocationType::Application,
///         "00000000-0000-0000-0000-000000000001",
///     ));
/// let middleware = PurviewAgentMiddleware::new(StaticTokenProvider::new("<graph-bearer-token>"), settings);
///
/// let agent = ChatAgent::builder(client)
///     .instructions("You are a helpful assistant.")
///     .middleware(Arc::new(middleware))
///     .build();
/// # let _ = agent;
/// # }
/// ```
pub struct PurviewAgentMiddleware(PurviewPolicyCore);

impl PurviewAgentMiddleware {
    pub fn new(token_provider: impl TokenProvider + 'static, settings: PurviewSettings) -> Self {
        Self(PurviewPolicyCore::new(token_provider, settings))
    }
}

#[async_trait]
impl Middleware<AgentContext> for PurviewAgentMiddleware {
    async fn process(
        &self,
        mut ctx: AgentContext,
        next: Next<AgentContext>,
    ) -> Result<AgentContext> {
        let (should_block, resolved_user_id) = self.0.check(&ctx.messages, None, "prompt").await?;
        if should_block {
            ctx.result = Some(AgentResponse {
                messages: vec![self.0.blocked_prompt_message()],
                ..Default::default()
            });
            ctx.terminate = true;
            return Ok(ctx);
        }

        let mut ctx = next.run(ctx).await?;

        if !ctx.is_streaming {
            let post_messages = ctx.result.as_ref().map(|r| r.messages.clone());
            if let Some(messages) = post_messages {
                let (should_block, _) = self
                    .0
                    .check(&messages, resolved_user_id.as_deref(), "response")
                    .await?;
                if should_block {
                    ctx.result = Some(AgentResponse {
                        messages: vec![self.0.blocked_response_message()],
                        ..Default::default()
                    });
                }
            }
        }
        Ok(ctx)
    }
}

/// Chat-client middleware variant of [`PurviewAgentMiddleware`], for
/// attaching Purview enforcement directly to a [`ChatClient`](agent_framework_core::client::ChatClient)
/// rather than an agent's middleware pipeline. Mirrors Python's
/// `PurviewChatPolicyMiddleware`; the policy logic is identical (see
/// the internal `PurviewPolicyCore`) — only the hook point (and result/message types)
/// differ.
pub struct PurviewChatMiddleware(PurviewPolicyCore);

impl PurviewChatMiddleware {
    pub fn new(token_provider: impl TokenProvider + 'static, settings: PurviewSettings) -> Self {
        Self(PurviewPolicyCore::new(token_provider, settings))
    }
}

#[async_trait]
impl Middleware<ChatContext> for PurviewChatMiddleware {
    async fn process(&self, mut ctx: ChatContext, next: Next<ChatContext>) -> Result<ChatContext> {
        let (should_block, resolved_user_id) = self.0.check(&ctx.messages, None, "prompt").await?;
        if should_block {
            ctx.result = Some(ChatResponse {
                messages: vec![self.0.blocked_prompt_message()],
                ..Default::default()
            });
            ctx.terminate = true;
            return Ok(ctx);
        }

        let mut ctx = next.run(ctx).await?;

        if !ctx.is_streaming {
            let post_messages = ctx.result.as_ref().map(|r| r.messages.clone());
            if let Some(messages) = post_messages {
                let (should_block, _) = self
                    .0
                    .check(&messages, resolved_user_id.as_deref(), "response")
                    .await?;
                if should_block {
                    ctx.result = Some(ChatResponse {
                        messages: vec![self.0.blocked_response_message()],
                        ..Default::default()
                    });
                }
            }
        }
        Ok(ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::StaticTokenProvider;
    use crate::settings::{PurviewAppLocation, PurviewLocationType};
    use agent_framework_core::middleware::{MiddlewarePipeline, Terminal};
    use agent_framework_core::tools::BoxFuture;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    fn valid_settings() -> PurviewSettings {
        PurviewSettings::new("Test App")
            .with_tenant_id("12345678-1234-1234-1234-123456789012")
            .with_purview_app_location(PurviewAppLocation::new(
                PurviewLocationType::Application,
                "app-1",
            ))
    }

    // These tests never reach the network: a missing tenant id / app
    // location fails deterministically before any HTTP call is attempted,
    // and "no resolvable user id" short-circuits to an allow verdict before
    // one too (see `ContentProcessor::evaluate`). That's enough to exercise
    // `PurviewAgentMiddleware`/`PurviewChatMiddleware::process`'s
    // short-circuit and error-propagation logic hermetically, using a "mock
    // next" continuation built from `agent-framework-core`'s own
    // `MiddlewarePipeline`/`Terminal` — the same machinery `ChatAgent` uses
    // internally — per this work package's "core test patterns" note. The
    // happy-path HTTP call itself (an actual `processContent` round trip) is
    // covered by `tests/loopback.rs`.

    fn agent_middleware(settings: PurviewSettings) -> PurviewAgentMiddleware {
        PurviewAgentMiddleware::new(StaticTokenProvider::new("token"), settings)
    }

    fn chat_middleware(settings: PurviewSettings) -> PurviewChatMiddleware {
        PurviewChatMiddleware::new(StaticTokenProvider::new("token"), settings)
    }

    fn agent_terminal(called: Arc<AtomicBool>, text: &'static str) -> Terminal<AgentContext> {
        Box::new(move |mut ctx: AgentContext| {
            called.store(true, Ordering::SeqCst);
            Box::pin(async move {
                ctx.result = Some(AgentResponse {
                    messages: vec![Message::assistant(text)],
                    ..Default::default()
                });
                Ok(ctx)
            }) as BoxFuture<Result<AgentContext>>
        })
    }

    fn chat_terminal(called: Arc<AtomicBool>, text: &'static str) -> Terminal<ChatContext> {
        Box::new(move |mut ctx: ChatContext| {
            called.store(true, Ordering::SeqCst);
            Box::pin(async move {
                ctx.result = Some(ChatResponse::from_text(text));
                Ok(ctx)
            }) as BoxFuture<Result<ChatContext>>
        })
    }

    /// `Result::unwrap_err` requires the `Ok` type to implement `Debug`,
    /// which `AgentContext`/`ChatContext` deliberately don't (they carry
    /// non-`Debug` middleware/result trait objects). Same shape, without
    /// that bound.
    fn expect_err<T>(result: Result<T>) -> agent_framework_core::error::Error {
        match result {
            Ok(_) => panic!("expected Err, got Ok"),
            Err(e) => e,
        }
    }

    // -- config-error propagation (no network: fails before any HTTP call) -

    #[tokio::test]
    async fn agent_middleware_propagates_config_error_without_ignore_exceptions() {
        let settings = PurviewSettings::new("Test App"); // no tenant_id/app_location
        let middleware = agent_middleware(settings);
        let pipeline = MiddlewarePipeline::new(vec![Arc::new(middleware)]);
        let called = Arc::new(AtomicBool::new(false));
        let ctx = AgentContext::new(vec![Message::user("hi")], false);

        let err = expect_err(
            pipeline
                .execute(ctx, agent_terminal(called.clone(), "should not be reached"))
                .await,
        );
        assert!(err.to_string().contains("tenant_id"));
        assert!(
            !called.load(Ordering::SeqCst),
            "next must not run on a pre-check error"
        );
    }

    #[tokio::test]
    async fn agent_middleware_ignore_exceptions_suppresses_config_error_and_allows() {
        let settings = PurviewSettings::new("Test App").with_ignore_exceptions(true);
        let middleware = agent_middleware(settings);
        let pipeline = MiddlewarePipeline::new(vec![Arc::new(middleware)]);
        let called = Arc::new(AtomicBool::new(false));
        let ctx = AgentContext::new(vec![Message::user("hi")], false);

        let result_ctx = pipeline
            .execute(ctx, agent_terminal(called.clone(), "real response"))
            .await
            .unwrap();
        assert!(
            called.load(Ordering::SeqCst),
            "next must run once the error is suppressed"
        );
        assert!(!result_ctx.terminate);
        assert_eq!(result_ctx.result.unwrap().text(), "real response");
    }

    // -- resolve_user_id short-circuits allow (no network) ------------------

    #[tokio::test]
    async fn agent_middleware_allows_when_no_resolvable_user_id() {
        // Valid settings, but the message carries no GUID-shaped user_id or
        // author_name -- resolve_user_id returns None, which the processor
        // treats as "cannot evaluate; allow" *without* an HTTP call. If it
        // did attempt one, this test would hang/fail trying to reach
        // graph.microsoft.com.
        let middleware = agent_middleware(valid_settings());
        let pipeline = MiddlewarePipeline::new(vec![Arc::new(middleware)]);
        let called = Arc::new(AtomicBool::new(false));
        let ctx = AgentContext::new(
            vec![Message::user("hello, nothing identifying here")],
            false,
        );

        let result_ctx = pipeline
            .execute(ctx, agent_terminal(called.clone(), "real response"))
            .await
            .unwrap();
        assert!(called.load(Ordering::SeqCst));
        assert!(!result_ctx.terminate);
        assert_eq!(result_ctx.result.unwrap().text(), "real response");
    }

    #[tokio::test]
    async fn agent_middleware_skips_post_check_when_streaming() {
        // is_streaming = true -> the response-phase check must never run,
        // so even a config-broken response phase can't surface an error.
        let middleware = agent_middleware(valid_settings());
        let pipeline = MiddlewarePipeline::new(vec![Arc::new(middleware)]);
        let called = Arc::new(AtomicBool::new(false));
        let ctx = AgentContext::new(vec![Message::user("hello, nothing identifying here")], true);

        let result_ctx = pipeline
            .execute(ctx, agent_terminal(called.clone(), "streamed response"))
            .await
            .unwrap();
        assert!(called.load(Ordering::SeqCst));
        assert_eq!(result_ctx.result.unwrap().text(), "streamed response");
    }

    #[tokio::test]
    async fn chat_middleware_allows_when_no_resolvable_user_id() {
        let middleware = chat_middleware(valid_settings());
        let pipeline = MiddlewarePipeline::new(vec![Arc::new(middleware)]);
        let called = Arc::new(AtomicBool::new(false));
        let ctx = ChatContext::new(
            vec![Message::user("hello, nothing identifying here")],
            agent_framework_core::types::ChatOptions::new(),
            false,
        );

        let result_ctx = pipeline
            .execute(ctx, chat_terminal(called.clone(), "real response"))
            .await
            .unwrap();
        assert!(called.load(Ordering::SeqCst));
        assert!(!result_ctx.terminate);
        assert_eq!(result_ctx.result.unwrap().text(), "real response");
    }

    #[tokio::test]
    async fn chat_middleware_propagates_config_error_without_ignore_exceptions() {
        let settings = PurviewSettings::new("Test App");
        let middleware = chat_middleware(settings);
        let pipeline = MiddlewarePipeline::new(vec![Arc::new(middleware)]);
        let called = Arc::new(AtomicBool::new(false));
        let ctx = ChatContext::new(
            vec![Message::user("hi")],
            agent_framework_core::types::ChatOptions::new(),
            false,
        );

        let err = expect_err(
            pipeline
                .execute(ctx, chat_terminal(called.clone(), "should not be reached"))
                .await,
        );
        assert!(err.to_string().contains("tenant_id"));
        assert!(!called.load(Ordering::SeqCst));
    }

    // -- blocked-message text/role ------------------------------------------

    #[test]
    fn blocked_messages_use_system_role_and_configured_text() {
        let core = PurviewPolicyCore::new(StaticTokenProvider::new("t"), valid_settings());
        let prompt_msg = core.blocked_prompt_message();
        let response_msg = core.blocked_response_message();
        assert_eq!(prompt_msg.role, Role::system());
        assert_eq!(prompt_msg.text(), "Prompt blocked by policy");
        assert_eq!(response_msg.role, Role::system());
        assert_eq!(response_msg.text(), "Response blocked by policy");
    }

    #[test]
    fn blocked_messages_honor_custom_text() {
        let settings = valid_settings()
            .with_blocked_prompt_message("custom prompt block")
            .with_blocked_response_message("custom response block");
        let core = PurviewPolicyCore::new(StaticTokenProvider::new("t"), settings);
        assert_eq!(core.blocked_prompt_message().text(), "custom prompt block");
        assert_eq!(
            core.blocked_response_message().text(),
            "custom response block"
        );
    }
}
