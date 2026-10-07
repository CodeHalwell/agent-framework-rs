//! AGENT-HOOKS-0.1 enforcement (experimental).
//!
//! Implements the [agent-hooks](https://github.com/responsibleai/agent-hooks)
//! control contract — a fixed set of interception points, the context a host
//! hands an interceptor at each, the `allow` / `deny` / `transform` verdict it
//! returns, and what the host must do with that verdict — as one coherent
//! feature on the framework's seams. Rust counterpart of upstream's
//! `agent_framework._agent_hooks` (Python, `ExperimentalFeature.AGENT_HOOKS`)
//! and `Microsoft.Agents.AI.AgentHooks` (.NET), whose statically typed shape
//! this follows.
//!
//! # Seams
//!
//! | Points | Seam |
//! | --- | --- |
//! | `agent_startup`, `input`, `agent_shutdown` | [`AgentHooksAgent`], a decorator around the [`Agent`](crate::agent::Agent) (like .NET's `AgentHooksAgent`) |
//! | `output` | agent middleware, first in the agent's list |
//! | `pre_model_call`, `post_model_call` | a chat-client decorator below the function-invocation loop (like .NET's `AgentHooksChatClient`), so each model service call is bracketed |
//! | `pre_tool_call`, `post_tool_call` | function middleware, first in the agent's list |
//!
//! The split follows the Rust agent's run order. Its agent middleware runs
//! *after* context providers have loaded history and *before* they persist
//! the run, so `output` lives there: a denied response is never persisted,
//! and a transformed one is persisted transformed. `input` must see only the
//! caller's request (not loaded history) and its transform must reach the
//! persisted history, so it runs in the decorator, before the inner run
//! starts. The agent's chat middleware wraps the whole tool loop rather than
//! each model call, so the model-call points use a chat-client decorator
//! instead.
//!
//! Install everything with [`AgentHooks::build_agent`] (or
//! [`AgentBuilder::build_with_agent_hooks`](crate::agent::AgentBuilder::build_with_agent_hooks)).
//! The seams are private and always installed together: installing only some
//! of them would enforce only part of the contract.
//!
//! ```no_run
//! use std::sync::Arc;
//! use agent_framework_core::agent_hooks::{interceptor_fn, AgentHooks, AgentHooksOptions, Verdict};
//! use agent_framework_core::prelude::*;
//!
//! # async fn demo(client: impl ChatClient + 'static) -> Result<()> {
//! let no_secrets = interceptor_fn(|ctx| async move {
//!     Ok(if ctx.target().to_string().contains("secret") {
//!         Verdict::deny("secret_detected")
//!     } else {
//!         Verdict::allow()
//!     })
//! });
//! let hooks = AgentHooks::new(AgentHooksOptions::new().interceptor(Arc::new(no_secrets)))?;
//! let agent = hooks.build_agent(Agent::builder(client).instructions("Be helpful."));
//! let response = agent.run(vec![Message::user("hello")], None).await?;
//! # let _ = response;
//! # Ok(())
//! # }
//! ```
//!
//! # Enforcement (fail closed)
//!
//! - Every point is emitted **before** its guarded action (pre points) or
//!   before its result is used (post points).
//! - A deny at `agent_startup`, `input`, `pre_model_call`, `post_model_call`
//!   or `output` fails the run with
//!   [`Error::InterceptionBlocked`](crate::Error::InterceptionBlocked). For a
//!   streaming run the error is returned by `run_stream` itself; no update is
//!   ever released.
//! - A deny at `pre_tool_call` / `post_tool_call` blocks that call (the tool
//!   does not run, or its result is discarded) and gives the model a
//!   tool-error payload naming the reason, so the loop continues (§6.2).
//! - A `host_error:*` deny anywhere — an interceptor that errors, panics or
//!   times out, an invalid verdict, an unappliable transform, zero
//!   interceptors — is a block. At the tool seam it halts the whole run via
//!   [`Error::MiddlewareFailure`](crate::Error::MiddlewareFailure), the
//!   function-invocation loop's one fail-closed escape, and the run fails
//!   with the recorded `InterceptionBlocked`.
//! - A transform is written back into the native value (messages, response,
//!   arguments, result) so the framework uses exactly what the interceptors
//!   approved. One that cannot be translated back fails the run with
//!   `Error::MiddlewareFailure` instead of being dropped. A transformed
//!   response drops its parsed structured `value`, which was derived from
//!   the pre-transform text.
//! - Streaming is fail-closed by buffering: model streams are assembled
//!   before `post_model_call`, and agent streams are produced only after
//!   `output` (spec §12.1a `buffered_output: true`).
//! - History: a run denied at any point fails, and context providers see the
//!   error and persist nothing. A transformed input or output is persisted
//!   transformed.
//! - A seam that finds no active run, or one owned by another bundle (for
//!   example nested guarded agents sharing a seam), fails closed.
//!
//! # Protocol layer
//!
//! Upstream depends on the external agent-hooks SDK for the emitter and
//! types. No Rust crate exists, so [`InterceptionEmitter`],
//! [`InterceptionContextBuilder`], [`Verdict`], [`Interceptor`] and
//! [`InterceptionRecord`] implement the subset the bundle needs: contexts
//! validated against the §4 envelope and the §12.3 size and depth bounds,
//! §5 verdict validation, `$target` transform paths (§5.2), the
//! `sequential/first_deny` profile with `on_approval: "stop"` (§7.4),
//! `enforce` and `evaluate_only` modes (§8), the §6.3 failure mapping, and
//! payload-free records (§10.3).
//!
//! # Not ported
//!
//! - **Approval seam (§9).** There is no approval resolver: a liftable deny
//!   (`Verdict::escalate`) is enforced as a plain deny, which the spec
//!   allows.
//! - **Composition profiles** other than `sequential/first_deny`
//!   (`sequential/run_all`, `parallel/strictest`, `parallel/unanimous`).
//! - **Identity providers (§10).** No `jcs-sha256` (RFC 8785 + SHA-256) or
//!   custom provider: records carry `null` identities and
//!   `identity_provider: null`, i.e. they are identity-unbound.
//! - **Cancellation.** Dropping a run's future emits no `agent_shutdown`
//!   (there is no async drop); `ShutdownReason::Cancelled` is never sent by
//!   the bundle.
//! - **Per-service-call history persistence** does not exist in the Rust
//!   agent, so there is nothing to gate there; history is persisted once per
//!   run, after `output`.
//! - **Hosted (provider-executed) tools** never pass the function seam.
//!   Their calls and results appear in the `post_model_call` content, where
//!   interceptors can deny or rewrite the response carrying them.
//! - **A client that runs its own tool loop.** The model-call seam wraps the
//!   client passed to [`Agent::builder`](crate::agent::Agent::builder). If
//!   that client already runs a tool loop (a `FunctionInvokingChatClient`),
//!   its tools execute below the model-call seam and outside the tool seam.
//!   .NET rejects such a client; the `ChatClient` trait offers no way to
//!   detect it here, so pass the raw provider client.
//! - **Agents nested inside the guarded run** that are not themselves
//!   guarded (e.g. a plain sub-agent used as a tool) are only covered at the
//!   tool seam that invokes them.

mod codecs;
mod protocol;
mod seams;

pub use protocol::{
    host_error, interceptor_fn, CompositionRecord, Decision, EmitOutcome, EnforcementMode,
    FnInterceptor, InterceptionBlocked, InterceptionContext, InterceptionContextBuilder,
    InterceptionEmitter, InterceptionPoint, InterceptionRecord, Interceptor, RecordSink,
    ShutdownReason, Transform, Verdict, VerdictSummary, Warning, COMPOSITION_PROFILE,
    DEFAULT_TIMEOUT, SPEC_VERSION,
};
pub use seams::{AgentHooks, AgentHooksAgent, AgentHooksOptions};

#[cfg(test)]
mod tests;
