//! # agent-framework-hosting
//!
//! Serve `agent-framework-rs` agents and workflows over HTTP. Three independent,
//! composable [`axum`] surfaces:
//!
//! - **DevUI-style API** ([`AgentHost`]) — entity discovery and
//!   OpenAI-Responses-flavored execution, mirroring the Python
//!   `agent_framework_devui` server:
//!   `GET /health`, `GET /v1/entities`, `GET /v1/entities/{id}/info`,
//!   `POST /v1/responses` (JSON or SSE), plus an embedded single-file debug
//!   page at `GET /` and `GET /ui`. See [`devui`].
//! - **A2A hosting** ([`a2a::A2ARouter`]) — the Agent-to-Agent protocol:
//!   `GET /.well-known/agent-card.json` and a JSON-RPC 2.0 `POST /`.
//! - **OpenAI Chat Completions** ([`openai_compat::OpenAiRouter`]) —
//!   `POST /v1/chat/completions` (JSON or SSE), for OpenAI-Chat clients.
//! - **AG-UI protocol** ([`agui::AgUiRouter`]) — CopilotKit's Agent-User
//!   Interaction protocol: `POST {path}` streaming camelCase SSE events
//!   (`RUN_STARTED` → `TEXT_MESSAGE_*` / `TOOL_CALL_*` → `RUN_FINISHED`),
//!   mirroring the Python `agent_framework_ag_ui` package.
//!
//! Each surface builds a plain [`axum::Router`] you can nest into your own app,
//! or run directly with [`AgentHost::serve`].
//!
//! ```no_run
//! use agent_framework_core::agent::ChatAgent;
//! use agent_framework_hosting::{AgentHost, a2a::A2ARouter, openai_compat::OpenAiRouter};
//!
//! # async fn demo(assistant: ChatAgent) -> std::io::Result<()> {
//! // DevUI host with one agent.
//! let host = AgentHost::new().agent("assistant", assistant.clone());
//!
//! // Compose the A2A and OpenAI surfaces alongside it.
//! let app = host
//!     .into_router()
//!     .merge(OpenAiRouter::for_agent("assistant", assistant.clone()).into_router())
//!     .nest(
//!         "/a2a",
//!         A2ARouter::for_agent("assistant", assistant, "http://localhost:8080/a2a").into_router(),
//!     );
//!
//! let listener = tokio::net::TcpListener::bind(("127.0.0.1", 8080)).await?;
//! axum::serve(listener, app).await
//! # }
//! ```
//!
//! ## Divergences from the reference
//! Per-surface divergences (stateless runs, streaming realized by run-to-
//! completion, metadata-derived A2A skills, omitted fields) are documented on
//! each module. The most consequential: **runs are stateless** — there is no
//! conversation store or workflow-resume endpoint, matching the work package's
//! decision that DevUI exposes no HTTP run-resume path of its own.

pub mod a2a;
pub mod agui;
pub mod devui;
pub mod openai_compat;
pub mod registry;

mod sse;
mod ui;
mod util;

pub use registry::{AgentHost, AgentRegistration, IntoAgentRegistration};

// Re-export the DevUI model types for callers building responses/clients.
pub use devui::models::{DiscoveryResponse, EntityInfo, HealthResponse};
