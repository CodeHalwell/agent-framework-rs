//! # agent-framework-declarative
//!
//! **Experimental:** everything here needs the
//! `experimental-declarative-agents` feature; without it the crate is empty.
//! See `docs/feature-stages.md`.
//!
//! Load [`Agent`](agent_framework_core::agent::Agent)s and
//! [`Workflow`](agent_framework_core::workflow::Workflow)s from declarative
//! YAML/JSON specifications, mirroring the Microsoft Agent Framework
//! `agent-framework-declarative` (Python) package.
//!
//! The crate is intentionally **provider-agnostic**: it never depends on the
//! OpenAI/Azure/Anthropic crates. Instead you register a
//! [`ChatClientFactory`] closure per provider string, a [`ToolRegistry`] of
//! native Rust tools, and (for workflows) an [`AgentRegistry`] of pre-built
//! agents, then call [`DeclarativeLoader::load_agent`] /
//! [`DeclarativeLoader::load_workflow`].
//!
//! ## SupportsAgentRun specs
//!
//! SupportsAgentRun specs follow the official schema vocabulary (`kind: Prompt`, `name`,
//! `instructions`, `model.id`/`provider`/`apiType`/`connection`/`options`,
//! `tools`, `outputSchema`, …). String fields support `${VAR}` /
//! `${VAR:-default}` environment interpolation.
//!
//! ## Workflow specs
//!
//! The upstream declarative *workflow* schema is a Power Platform / Copilot
//! Studio imperative DSL that does not map onto this port's graph engine. This
//! crate therefore defines a documented Rust-native [`WorkflowSpec`] that drives
//! the existing `WorkflowBuilder` and orchestration builders — either via
//! orchestration shorthand (`type: sequential | concurrent | group_chat |
//! handoff`) or an explicit node/edge graph. See [`workflow`] for details.
//!
//! ## Example
//!
//! ```no_run
//! # #[cfg(feature = "experimental-declarative-agents")]
//! # mod example {
//! use std::sync::Arc;
//! use agent_framework_core::prelude::*;
//! use agent_framework_declarative::{ChatClientFactory, DeclarativeLoader};
//!
//! # fn make_client() -> Arc<dyn ChatClient> { unimplemented!() }
//! # async fn demo() -> Result<()> {
//! let loader = DeclarativeLoader::new().with_client_factory(
//!     ChatClientFactory::new().with("OpenAI.Chat", |_model| Ok(make_client())),
//! );
//!
//! let yaml = r#"
//! kind: Prompt
//! name: Assistant
//! instructions: You are a helpful assistant.
//! model:
//!   id: gpt-4.1-mini
//!   provider: OpenAI
//!   apiType: Chat
//!   options:
//!     temperature: 0.7
//! "#;
//!
//! let agent = loader.load_agent(yaml).unwrap();
//! let response = agent.run_once("Hello!").await?;
//! println!("{}", response.text());
//! # Ok(())
//! # }
//! # }
//! ```

#![warn(missing_docs)]
// Upstream marks declarative agents experimental (Python `DECLARATIVE_AGENTS`),
// so the whole crate is behind a feature and is empty without it. See
// docs/feature-stages.md.
#![cfg(feature = "experimental-declarative-agents")]
#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg_attr(docsrs, doc(cfg(feature = "experimental-declarative-agents")))]
pub mod agent;
#[cfg_attr(docsrs, doc(cfg(feature = "experimental-declarative-agents")))]
pub mod condition;
#[cfg_attr(docsrs, doc(cfg(feature = "experimental-declarative-agents")))]
pub mod env;
#[cfg_attr(docsrs, doc(cfg(feature = "experimental-declarative-agents")))]
pub mod error;
#[cfg_attr(docsrs, doc(cfg(feature = "experimental-declarative-agents")))]
pub mod loader;
#[cfg_attr(docsrs, doc(cfg(feature = "experimental-declarative-agents")))]
pub mod registry;
#[cfg_attr(docsrs, doc(cfg(feature = "experimental-declarative-agents")))]
pub mod workflow;

#[cfg_attr(docsrs, doc(cfg(feature = "experimental-declarative-agents")))]
pub use agent::{
    AgentSpec, ApprovalModeDetail, ApprovalModeSpec, ConnectionSpec, ModelOptions, ModelSpec,
    PropertySchema, PropertySpec, ToolSpec,
};
#[cfg_attr(docsrs, doc(cfg(feature = "experimental-declarative-agents")))]
pub use env::{EnvSource, ProcessEnv};
#[cfg_attr(docsrs, doc(cfg(feature = "experimental-declarative-agents")))]
pub use error::{DeclarativeError, Result};
#[cfg_attr(docsrs, doc(cfg(feature = "experimental-declarative-agents")))]
pub use loader::DeclarativeLoader;
#[cfg_attr(docsrs, doc(cfg(feature = "experimental-declarative-agents")))]
pub use registry::{
    AgentRegistry, ChatClientFactory, ClientFactoryResult, FactoryError, PredicateRegistry,
    ToolRegistry,
};
#[cfg_attr(docsrs, doc(cfg(feature = "experimental-declarative-agents")))]
pub use workflow::{
    CaseSpec, EdgeSpec, FanInSpec, FanOutSpec, HandoffEdgeSpec, NodeSpec, OrchestrationType,
    SwitchSpec, WorkflowSpec,
};
