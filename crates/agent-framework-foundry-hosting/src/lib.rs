//! Host agents and workflows as Azure AI Foundry hosted agents.
//!
//! Port of upstream `agent_framework_foundry_hosting` (Python) and
//! `Microsoft.Agents.AI.Foundry.Hosting` (.NET), including the parts of the
//! `azure-ai-agentserver-*` container protocol they build on:
//!
//! - [`FoundryHostConfig`]: the platform's `FOUNDRY_*` / `PORT` variables.
//! - [`ResponsesHostServer`]: an agent or workflow on `POST /responses`
//!   (sync, streamed, background with stream replay; `previous_response_id`
//!   and `conversation` continuation; history sources; rich output items;
//!   checkpoint-resumed workflow pauses; get, delete, cancel, input items),
//!   with [`InMemoryResponseStore`] or [`FileResponseStore`].
//! - `InvocationsHostServer` (the `session-store` feature): the
//!   Invocations protocol, with durable per-session agent state.
//! - [`FoundryStorage`] and the stores over it: [`FoundryCheckpointStore`],
//!   [`FoundryFunctionApprovalStore`], and (the `session-store` feature,
//!   since core's session store is experimental) `FoundryAgentSessionStore`.
//! - [`FoundryToolbox`]: a Foundry toolbox's MCP endpoint as a tool source.
//! - Every server adds `GET /readiness`, the snapshot lifecycle routes
//!   (`after-restore` binds the session and applies its environment
//!   overrides), a request span joining the caller's `traceparent`, and
//!   `x-request-id` / `x-platform-server` headers ([`with_host_routes`]).
//!   The Invocations server also serves registered OpenAPI / AsyncAPI
//!   documents.
//!
//! ```no_run
//! use std::sync::Arc;
//! use agent_framework_core::prelude::*;
//! use agent_framework_foundry_hosting::{FoundryHostConfig, ResponsesHostServer};
//!
//! # async fn demo(client: impl ChatClient + 'static) -> Result<()> {
//! let agent = Arc::new(Agent::builder(client).instructions("Be helpful.").build());
//! ResponsesHostServer::agent(agent)
//!     .with_config(FoundryHostConfig::from_env()?)
//!     .serve()
//!     .await
//! # }
//! ```
//!
//! # Divergences
//!
//! - **Responses output** covers text (with URL, file, file-path and
//!   container-file citations), refusals, reasoning, function calls and
//!   results, MCP calls, image generation, shell calls, approval and OAuth
//!   consent requests, and usage, which is what upstream projects. Not
//!   projected: computer-use items (core has no computer-use content; that
//!   is upstream's experimental `COMPUTER_USE` feature). Container files an
//!   MCP tool result announces in its `_meta` (`container_file_citations`)
//!   become `container_file_citation` annotations on later output text that
//!   names them (they are not carried across a restart, which upstream does
//!   only for its resilient runtime).
//!   Function-approval requests round-trip: each `mcp_approval_request` is
//!   saved (to the configured `FunctionApprovalStoreProvider`, else in
//!   process) and a later `mcp_approval_response` resumes the call.
//! - **History.** With the `session-store` feature, the agent sessions of
//!   `HistorySource::Agent` / `Service` (history included) go through a
//!   session store (`with_session_store_provider`, e.g. Foundry storage),
//!   so continuations survive a restart; without it they are kept live in
//!   process. Provider-native background polling (`background_source =
//!   "provider"`) is not ported.
//! - **Workflow pauses** answer only `function_call_output` replies;
//!   approval (`mcp_approval_response`) and computer-call replies to a
//!   paused workflow are not ported, and a typed reply is passed as JSON
//!   when its output parses as JSON (the Rust request carries no response
//!   type).
//! - **Local storage** defaults to in memory; [`FileResponseStore`] gives
//!   the agentserver's file-backed store (not partitioned by user).
//! - **Toolbox** tokens are fetched from the credential for every MCP
//!   request. Its skills (feature `skills`) include `archive` (ZIP) skills,
//!   unpacked in memory; `skills_provider` discovers once, and
//!   `ToolboxSkillsProvider` discovers lazily, caches, and refreshes (see
//!   `agent_framework_mcp::skills` and [`toolbox::ToolboxSkillsProvider`]).
//! - **Restarts**: background responses a previous process left `queued`
//!   or `in_progress` are failed by
//!   [`ResponsesHostServer::fail_interrupted_responses`] (run by `serve`),
//!   with upstream's recovery error. Not ported: the agentserver's resilient
//!   task runtime (built on the Foundry Tasks service), which can resume a
//!   workflow from its checkpoint, and provider-native background polling.
//! - **WebSocket invocations** (feature `websocket`): [`invocations_ws`]
//!   serves `/invocations_ws`.
//! - **Voice invocations** (feature `voice`, implies `websocket`):
//!   [`voice`] serves upstream's experimental typed Voice Live Bridge relay
//!   (`VoiceAgentServerHost`) on `/voice_conversation_engine` and
//!   `/invocations_ws`. Not ported: the `x-agent-original-subject` header
//!   check.
//! - **Telemetry** (feature `otel`): caller trace context and W3C baggage,
//!   exporter selection, and Foundry span enrichment
//!   ([`FoundryEnrichmentSpanProcessor`]); the agentserver's OTel distro is
//!   otherwise not ported.

#[cfg(feature = "experimental-computer-use")]
mod computer;
pub mod config;
pub mod file_store;
pub mod host;
#[cfg(feature = "session-store")]
#[cfg_attr(docsrs, doc(cfg(feature = "session-store")))]
pub mod invocations;
#[cfg(feature = "websocket")]
#[cfg_attr(docsrs, doc(cfg(feature = "websocket")))]
pub mod invocations_ws;
mod output;
pub mod responses;
pub mod scope;
#[cfg(feature = "session-store")]
#[cfg_attr(docsrs, doc(cfg(feature = "session-store")))]
pub mod sessions;
pub mod storage;
pub mod stores;
pub mod toolbox;
#[cfg(feature = "voice")]
#[cfg_attr(docsrs, doc(cfg(feature = "voice")))]
pub mod voice;

pub use config::FoundryHostConfig;
pub use file_store::FileResponseStore;
#[cfg(feature = "otel")]
pub use host::{configure_observability, FoundryEnrichmentSpanProcessor};
pub use host::{
    parse_traceparent, serve, with_host_routes, AfterRestoreHook, AgentSessionContext, HostOptions,
};
#[cfg(feature = "session-store")]
pub use invocations::{InvocationRequest, InvocationRun, InvocationsHostServer, WorkflowTurn};
pub use responses::{
    BackgroundSource, HistorySource, HostedResponseRequest, InMemoryResponseStore, OptionsHook,
    ResponseStore, ResponsesHostServer, StoredResponse, WorkflowPause,
};
pub use scope::{
    FoundryOriginalSubject, FoundryRequestContext, FoundryRequestScope, InvalidOriginalSubject,
};
#[cfg(feature = "session-store")]
pub use sessions::{
    FoundryAgentSessionStore, FoundrySessionStoreProvider, InMemorySessionStoreProvider,
    SessionStoreProvider,
};
pub use storage::{FoundryStateStore, FoundryStorage, FoundryStorageEndpoint, StorageError};
pub use stores::{
    CheckpointStoreProvider, FoundryCheckpointStore, FoundryFunctionApprovalStore,
    FunctionApprovalStore, FunctionApprovalStoreProvider, InMemoryFunctionApprovalStore,
};
pub use toolbox::FoundryToolbox;
#[cfg(feature = "skills")]
pub use toolbox::{ToolboxSkillsOptions, ToolboxSkillsProvider};
