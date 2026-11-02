//! Host agents and workflows as Azure AI Foundry hosted agents.
//!
//! Port of upstream `agent_framework_foundry_hosting` (Python) and
//! `Microsoft.Agents.AI.Foundry.Hosting` (.NET), including the parts of the
//! `azure-ai-agentserver-*` container protocol they build on:
//!
//! - [`FoundryHostConfig`]: the platform's `FOUNDRY_*` / `PORT` variables.
//! - [`ResponsesHostServer`]: an agent or workflow on `POST /responses`
//!   (sync, streamed, background; `previous_response_id` and
//!   `conversation` continuation; get, delete, cancel, input items).
//! - `InvocationsHostServer` (the `session-store` feature): the
//!   Invocations protocol, with durable per-session agent state.
//! - [`FoundryStorage`] and the stores over it: [`FoundryCheckpointStore`],
//!   [`FoundryFunctionApprovalStore`], and (the `session-store` feature,
//!   since core's session store is experimental) `FoundryAgentSessionStore`.
//! - [`FoundryToolbox`]: a Foundry toolbox's MCP endpoint as a tool source.
//! - Every server adds `GET /readiness`, the snapshot lifecycle routes, and
//!   `x-request-id` / `x-platform-server` headers ([`with_host_routes`]).
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
//! - **History.** The Responses server implements upstream's default
//!   `history_source="agent_server"` only (the host replays the stored
//!   transcript); `"service"` / `"agent"` sources, provider-native background
//!   polling, OAuth consent links, and `prepare_options` are not ported.
//! - **Responses output.** Text messages and declared (client-side) function
//!   calls are emitted; reasoning, refusal, MCP, file/web-search, image and
//!   computer-use items, annotations and usage streaming are not. Streamed
//!   responses cannot be replayed with `GET …?stream=true`.
//! - **Workflows** run fresh per request: checkpoint-backed continuation
//!   (resuming `request_info` pauses on a later turn) is not ported.
//! - **Local storage** defaults to in memory (upstream: files).
//! - **Toolbox** tokens are fetched when the tool is built, not per request;
//!   the toolbox skills provider is not ported.
//! - The agentserver's resilient task runtime, WebSocket/voice
//!   invocations, OpenAPI/AsyncAPI spec routes, and tracing are not ported.

pub mod config;
pub mod host;
#[cfg(feature = "session-store")]
#[cfg_attr(docsrs, doc(cfg(feature = "session-store")))]
pub mod invocations;
pub mod responses;
pub mod scope;
#[cfg(feature = "session-store")]
#[cfg_attr(docsrs, doc(cfg(feature = "session-store")))]
pub mod sessions;
pub mod storage;
pub mod stores;
pub mod toolbox;

pub use config::FoundryHostConfig;
pub use host::{serve, with_host_routes, HostOptions};
#[cfg(feature = "session-store")]
pub use invocations::{InvocationRequest, InvocationRun, InvocationsHostServer, WorkflowTurn};
pub use responses::{
    HostedResponseRequest, InMemoryResponseStore, ResponseStore, ResponsesHostServer,
    StoredResponse,
};
pub use scope::{FoundryRequestContext, FoundryRequestScope};
#[cfg(feature = "session-store")]
pub use sessions::{
    FoundryAgentSessionStore, FoundrySessionStoreProvider, InMemorySessionStoreProvider,
    SessionStoreProvider,
};
pub use storage::{FoundryStateStore, FoundryStorage, FoundryStorageEndpoint, StorageError};
pub use stores::{
    CheckpointStoreProvider, FoundryCheckpointStore, FoundryFunctionApprovalStore,
    FunctionApprovalStore, FunctionApprovalStoreProvider,
};
pub use toolbox::FoundryToolbox;
