//! # agent-framework-copilotstudio
//!
//! A [Microsoft Copilot Studio](https://copilotstudio.microsoft.com/) agent
//! **client** for `agent-framework-rs`: talk to a published (or prebuilt)
//! Copilot Studio agent over its Direct-to-Engine (D2E) API as if it were a
//! local [`SupportsAgentRun`](agent_framework_core::agent::SupportsAgentRun).
//!
//! This is the Rust equivalent of `agent_framework_copilotstudio`
//! (`CopilotStudioAgent`) in the Python reference implementation. Python
//! itself does not speak the D2E wire protocol directly — it wraps the
//! separate `microsoft-agents-copilotstudio-client` PyPI package
//! (`microsoft_agents.copilotstudio.client.CopilotClient`), which is not
//! part of this repository and has no Rust equivalent to wrap. This crate
//! therefore speaks the D2E wire protocol itself.
//!
//! ## Fidelity: how the wire protocol was determined
//!
//! Per this work package's constraints, no network access (`WebFetch`,
//! `WebSearch`, or MCP documentation tools) was used to research the D2E
//! protocol or the `microsoft-agents-copilotstudio-client` package. Its
//! actual source (version 1.1.0) happened to already be present on local
//! disk (a prior session's package download, sitting unrelated to this repo)
//! and was read directly — the same way any other local file was read.
//! URL construction, request/response shapes, and headers below are a
//! faithful, line-by-line port of that package's
//! `power_platform_environment.py` / `connection_settings.py` /
//! `copilot_client.py` / `execute_turn_request.py`, cross-checked against
//! the Python wrapper package's tests
//! (`python/packages/copilotstudio/tests/`) in the reference repo at
//! `/home/user/agent-framework`. Where this module's docs say "mirrors" a
//! specific Python source line, that is a **high-fidelity** claim, not an
//! inferred convention. The one deliberately-out-of-scope wire feature is
//! `tasks/subscribe` (`CopilotClient.subscribe`) — documented Python-side as
//! "for MSFT internal use only" — which this crate does not implement.
//!
//! ## Layout
//!
//! - [`settings`] — [`CopilotStudioSettings`] (`COPILOTSTUDIOAGENT__*` env
//!   vars), [`settings::PowerPlatformCloud`],
//!   [`settings::AgentType`], and
//!   [`CopilotStudioConnectionSettings`] (Direct-to-Engine conversation URL
//!   construction).
//! - [`auth`] — [`TokenProvider`]: bring-your-own bearer token (see "Auth
//!   burden" below).
//! - [`activity`] — the Direct-to-Engine `Activity` wire shape and
//!   SSE/JSON-array response parsing.
//! - [`agent`] — [`CopilotStudioAgent`]: the
//!   [`SupportsAgentRun`](agent_framework_core::agent::SupportsAgentRun) wrapper.
//!
//! ## Example
//!
//! ```no_run
//! use agent_framework_copilotstudio::{
//!     CopilotStudioAgent, CopilotStudioConnectionSettings, CopilotStudioSettings,
//!     StaticTokenProvider,
//! };
//! use agent_framework_core::prelude::*;
//!
//! # async fn demo() -> Result<()> {
//! // Reads COPILOTSTUDIOAGENT__ENVIRONMENTID / _SCHEMANAME (both required).
//! let settings = CopilotStudioSettings::from_env();
//! let connection = CopilotStudioConnectionSettings::from_settings(&settings)?;
//!
//! // Bring your own Power Platform API bearer token (see the `auth` module
//! // docs for why this port doesn't acquire one for you).
//! let token_provider = StaticTokenProvider::new("<power-platform-api-bearer-token>");
//!
//! let agent = CopilotStudioAgent::new(connection, token_provider).with_name("my-copilot");
//! let response = agent.run_once("What is the capital of France?").await?;
//! println!("{}", response.text());
//! # Ok(())
//! # }
//! ```
//!
//! Reuse the same [`AgentThread`](agent_framework_core::threads::AgentThread)
//! across calls for a multi-turn conversation:
//!
//! ```no_run
//! # use agent_framework_copilotstudio::{CopilotStudioAgent, CopilotStudioConnectionSettings, StaticTokenProvider};
//! use agent_framework_core::prelude::*;
//!
//! # async fn demo() -> Result<()> {
//! # let agent = CopilotStudioAgent::new(
//! #     CopilotStudioConnectionSettings::new("env-id", "schema-name"),
//! #     StaticTokenProvider::new("token"),
//! # );
//! let mut thread = agent.get_new_thread();
//! agent.run(vec![Message::user("What's the weather in Seattle?")], Some(&mut thread)).await?;
//! // The Direct-to-Engine conversation id is now attached to `thread`, so
//! // this second call continues the same conversation.
//! let reply = agent.run(vec![Message::user("What about tomorrow?")], Some(&mut thread)).await?;
//! println!("{}", reply.text());
//! # Ok(())
//! # }
//! ```
//!
//! ## Auth burden
//!
//! The Python reference acquires its own token via MSAL
//! (`msal.PublicClientApplication`, `agent_framework_copilotstudio.acquire_token`):
//! silent/cached-account first, falling back to an *interactive
//! browser-popup login*. That flow is not something this crate can or should
//! reproduce headlessly. Instead, [`TokenProvider`] pushes token acquisition
//! entirely onto the caller — wrap an MSAL confidential-client / managed-identity
//! flow performed elsewhere, or use [`StaticTokenProvider`] for a
//! pre-fetched/test token. The required scope is
//! `https://api.powerplatform.com/.default` (or the equivalent for a
//! non-Prod [`settings::PowerPlatformCloud`]).
//!
//! ## Divergences from the Python reference
//!
//! - **No `microsoft-agents-copilotstudio-client` dependency.** This crate
//!   speaks Direct-to-Engine directly over `reqwest`, as described above.
//! - **Real conversation-id continuity.** Python's `CopilotStudioAgent.run`
//!   calls `self._start_new_conversation()` *unconditionally* on every
//!   invocation — even when a `thread` with an existing
//!   `service_thread_id` is passed in — discarding whatever conversation
//!   context existed before. This port instead starts a new Direct-to-Engine
//!   conversation only the first time an [`AgentThread`](agent_framework_core::threads::AgentThread)
//!   is used, and reuses its conversation id on subsequent calls (the same
//!   fix `agent-framework-a2a`'s `A2AAgent` applies for `contextId`/`taskId`
//!   continuity — see that crate's docs for the identical rationale).
//! - **Only the newest message is sent.** Consistent with adding real
//!   continuity above: this port sends just the last message's text as the
//!   outgoing `message` activity, relying on the conversation id for
//!   everything earlier — mirroring `agent-framework-a2a`. Python instead
//!   joins *every* `Message` passed to a single `run()` call with `"\n"`
//!   into one question string (relevant when a caller passes several
//!   messages in one call, e.g. `agent.run(["Hello", "How are you?"])`);
//!   that per-call flattening is not reproduced here.
//! - **`run_stream` uses the trait's buffered default.**
//!   [`agent_framework_core::agent::SupportsAgentRun`] now has an object-safe `run_stream`
//!   (with a default that runs to completion and replays the messages as
//!   updates), but [`CopilotStudioAgent`] deliberately does **not** override it
//!   with real streaming. This sidesteps a genuine oddity in the Python
//!   reference: `run_stream`'s `_process_activities(activities,
//!   streaming=True)` call only ever surfaces `type == "typing"` activity text
//!   as updates and *never* yields the final `type == "message"` activity — so
//!   real Copilot Studio streaming would emit only interim typing indicators.
//!   This port's [`SupportsAgentRun::run`](agent_framework_core::agent::SupportsAgentRun::run)
//!   mirrors Python's **non**-streaming path (`streaming=False`), which
//!   correctly surfaces `message` activities and skips `typing`/other types;
//!   the buffered `run_stream` default then replays that complete answer.
//! - **Response parsing accepts a bare JSON array**, not just SSE `event:
//!   activity` / `data:` frames — see [`activity::parse_activities`] for
//!   why.
//! - **Not implemented**: `CopilotClient.subscribe` (`tasks/.../subscribe`,
//!   documented upstream as internal-use-only), the experimental
//!   "island" endpoint header (`x-ms-d2e-experimental`), and
//!   `start_conversation_with_request`'s custom `StartRequest` (only the
//!   `emitStartConversationEvent` body `CopilotStudioAgent` itself sends is
//!   reproduced).

pub mod activity;
pub mod agent;
pub mod auth;
pub mod settings;

pub use agent::CopilotStudioAgent;
pub use auth::{StaticTokenProvider, TokenProvider};
pub use settings::{
    AgentType, CopilotStudioConnectionSettings, CopilotStudioSettings, PowerPlatformCloud,
    API_VERSION,
};
