//! # agent-framework-a2a
//!
//! An [Agent2Agent (A2A)](https://a2a-protocol.org/) protocol **client** for
//! `agent-framework-rs`: talk to any A2A-compliant remote agent as if it were
//! a local [`Agent`](agent_framework_core::agent::Agent).
//!
//! This is the Rust equivalent of `agent_framework_a2a` (`A2AAgent`) in the
//! Python reference implementation, built directly against the A2A JSON-RPC
//! 2.0 wire protocol rather than wrapping the `a2a-sdk` Python package —
//! there is no Rust equivalent of that SDK to wrap, so this crate speaks the
//! wire protocol itself.
//!
//! ## Layout
//!
//! - [`types`] — spec-faithful wire types: [`AgentCard`], [`Message`],
//!   [`Part`] (`Text` | `File` | `Data`), [`Task`], [`TaskState`], and the
//!   JSON-RPC parameter/result shapes.
//! - [`client`] — [`A2AClient`]: JSON-RPC 2.0 over HTTP POST
//!   (`message/send`, `message/stream`, `tasks/get`, `tasks/cancel`,
//!   `tasks/resubscribe`, `tasks/pushNotificationConfig/{set,get}`), plus
//!   [`AgentCard`] discovery via `.well-known` (auto-upgrading to the
//!   authenticated extended card when the server advertises it).
//! - [`agent`] — [`A2AAgent`]: the [`Agent`](agent_framework_core::agent::Agent)
//!   wrapper, converting [`ChatMessage`](agent_framework_core::types::ChatMessage)s
//!   to/from A2A [`Message`]s and [`Task`]s.
//!
//! ## Example
//!
//! ```no_run
//! use agent_framework_a2a::A2AAgent;
//! use agent_framework_core::prelude::*;
//!
//! # async fn demo() -> Result<()> {
//! let agent = A2AAgent::from_url("weather-agent", "https://weather.example.com/a2a");
//! let response = agent.run_once("What's the forecast for Seattle?").await?;
//! println!("{}", response.text());
//! # Ok(())
//! # }
//! ```
//!
//! Reuse the same [`AgentThread`](agent_framework_core::threads::AgentThread)
//! across calls for a multi-turn conversation:
//!
//! ```no_run
//! use agent_framework_a2a::A2AAgent;
//! use agent_framework_core::prelude::*;
//!
//! # async fn demo() -> Result<()> {
//! let agent = A2AAgent::from_url("weather-agent", "https://weather.example.com/a2a");
//! let mut thread = agent.get_new_thread();
//! agent.run(vec![ChatMessage::user("What's the forecast for Seattle?")], Some(&mut thread)).await?;
//! // The remote agent's contextId/taskId are now attached to `thread`, so
//! // this second call continues the same A2A conversation.
//! let reply = agent.run(vec![ChatMessage::user("What about tomorrow?")], Some(&mut thread)).await?;
//! println!("{}", reply.text());
//! # Ok(())
//! # }
//! ```
//!
//! ## Divergences from the Python reference
//!
//! - **No `a2a-sdk` dependency, no transport negotiation.** The Python
//!   package wraps `a2a-sdk`'s `Client`/`ClientFactory`, which can speak
//!   JSON-RPC, gRPC, or REST and negotiates a transport from the
//!   [`AgentCard`]. This crate only speaks JSON-RPC 2.0 over HTTP — the
//!   transport every A2A server is required to support — so there is no
//!   negotiation step.
//! - **`contextId`/`taskId` continuity is new.** The Python `A2AAgent`
//!   accepts a `thread` parameter on `run`/`run_stream` but never actually
//!   reads or writes it — every call sends a context-less message, leaving
//!   multi-turn continuity entirely up to whatever session affinity the
//!   remote agent infers on its own. This port stores the last response's
//!   `contextId`/`taskId` in
//!   [`AgentThread::service_thread_id`](agent_framework_core::threads::AgentThread)
//!   (JSON-encoded, since that field is a single string) and replays them on
//!   the same thread's next [`A2AAgent::run`](agent_framework_core::agent::Agent::run)
//!   call, so a real multi-turn conversation works as long as the same
//!   thread is reused (see the second example above) and doesn't already
//!   have a local message store attached (e.g. one borrowed from a
//!   `ChatAgent`) — continuity is skipped, not an error, in that case.
//! - **`input-required` surfaces the agent's question.** If a [`Task`] comes
//!   back in [`TaskState::InputRequired`], this crate returns
//!   `task.status.message` as the response text, so the caller has
//!   something to show/act on. The Python reference has no special case for
//!   this state and would silently produce no messages unless the server
//!   happens to also put that message in `history`.
//! - **No polling loop.** [`A2AAgent::run`](agent_framework_core::agent::Agent::run)
//!   performs exactly one `message/send` call and maps whatever comes back,
//!   including a still-`working`/`submitted` [`Task`] (which maps to zero
//!   messages, with the task id as `response_id`). It does not poll
//!   `tasks/get` until the task reaches a terminal state — call
//!   [`A2AClient::get_task`] directly (via [`A2AAgent::client`]) if you need
//!   that.
//! - **Streaming (`message/stream`) is implemented on [`A2AClient`]**, as
//!   [`A2AClient::send_message_stream`], returning the raw sequence of
//!   `Message` / `Task` / status-update / artifact-update events over SSE.
//!   It is not wired into [`A2AAgent`] — `run` always uses the non-streaming
//!   `message/send` — because
//!   [`agent_framework_core::agent::Agent`] has no `run_stream` requirement
//!   (unlike Python's `BaseAgent`), so there's nothing in the trait for a
//!   streamed run to plug into.
//! - **Push notifications** (`tasks/pushNotificationConfig/set` / `/get`):
//!   [`A2AClient::set_push_notification_config`] /
//!   [`A2AClient::get_push_notification_config`]. Note the `get` request's
//!   params shape genuinely differs from `set`'s on the wire (the task id is
//!   sent under `id`, not `taskId`) — a real A2A 0.3.0 spec/SDK
//!   inconsistency, faithfully preserved here rather than "fixed".
//! - **`tasks/resubscribe`**: [`A2AClient::resubscribe`] reconnects to an
//!   existing task's event stream, returning the exact same
//!   [`A2AEventStream`] shape as [`A2AClient::send_message_stream`] (both
//!   share one response type on the wire, and one implementation here).
//! - **Authenticated extended card**: when a discovered [`AgentCard`] sets
//!   `supportsAuthenticatedExtendedCard`, [`A2AClient::get_agent_card`]
//!   automatically calls [`A2AClient::get_extended_card`]
//!   (`agent/getAuthenticatedExtendedCard`) and upgrades to it, falling back
//!   to the base card on failure. This upgrade only happens on the
//!   `.well-known` discovery path — [`A2AClient::from_card`] never performs
//!   it, preserving that constructor's "no discovery call is ever made"
//!   contract.
//! - **Not implemented**: the *serving* side of any of the above
//!   (`agent-framework-hosting`'s A2A router doesn't expose push
//!   notification config, resubscribe, or an extended card endpoint) — this
//!   crate is a client only, matching its scope.

pub mod agent;
pub mod client;
pub mod protocol;
pub mod types;

pub use agent::A2AAgent;
pub use client::{A2AClient, A2AEventStream};
pub use types::{
    AgentCapabilities, AgentCard, AgentProvider, AgentSkill, Artifact, DataPart, FileData,
    FilePart, FileWithBytes, FileWithUri, Message, MessageRole, MessageSendConfiguration,
    MessageSendParams, MessageStreamEvent, Part, PushNotificationAuthenticationInfo,
    PushNotificationConfig, SendMessageResult, Task, TaskArtifactUpdateEvent,
    TaskPushNotificationConfig, TaskState, TaskStatus, TaskStatusUpdateEvent, TextPart,
};
