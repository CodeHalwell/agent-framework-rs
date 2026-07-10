//! The [`AgentHost`] registry: agents and workflows exposed over HTTP.
//!
//! Entities are keyed by a caller-chosen `name` that doubles as the routing
//! `id` used in URLs (`/v1/entities/{id}/…`) and in `/v1/responses`. Display
//! metadata (`name`, `description`, `instructions`) is captured at registration
//! time from the concrete type where it is cheaply available; see
//! [`IntoAgentRegistration`].

use std::collections::HashMap;
use std::sync::Arc;

use agent_framework_core::agent::{Agent, ChatAgent};
use agent_framework_core::workflow::{Workflow, WorkflowAgent};

/// An agent plus the display metadata captured from its concrete type.
///
/// Produced by [`IntoAgentRegistration`]. `description` and `instructions` are
/// best-effort: they are populated for [`ChatAgent`]/[`WorkflowAgent`] (whose
/// accessors are public) and left `None` for an opaque `Arc<dyn Agent>`.
pub struct AgentRegistration {
    pub(crate) agent: Arc<dyn Agent>,
    pub(crate) description: Option<String>,
    pub(crate) instructions: Option<String>,
}

impl AgentRegistration {
    /// Build a registration from a bare agent handle with no extra metadata.
    pub fn new(agent: Arc<dyn Agent>) -> Self {
        Self {
            agent,
            description: None,
            instructions: None,
        }
    }

    /// Set the description (builder style).
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the instructions (builder style).
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }
}

/// Conversion into an [`AgentRegistration`], letting [`AgentHost::agent`] accept
/// a [`ChatAgent`], a [`WorkflowAgent`], or a bare `Arc<dyn Agent>` naturally
/// while still capturing description/instructions from the concrete types.
pub trait IntoAgentRegistration {
    fn into_agent_registration(self) -> AgentRegistration;
}

impl IntoAgentRegistration for AgentRegistration {
    fn into_agent_registration(self) -> AgentRegistration {
        self
    }
}

impl IntoAgentRegistration for ChatAgent {
    fn into_agent_registration(self) -> AgentRegistration {
        let description = self.description().map(str::to_string);
        let instructions = self.instructions().map(str::to_string);
        AgentRegistration {
            agent: Arc::new(self),
            description,
            instructions,
        }
    }
}

impl IntoAgentRegistration for Arc<ChatAgent> {
    fn into_agent_registration(self) -> AgentRegistration {
        let description = self.description().map(str::to_string);
        let instructions = self.instructions().map(str::to_string);
        AgentRegistration {
            agent: self,
            description,
            instructions,
        }
    }
}

impl IntoAgentRegistration for WorkflowAgent {
    fn into_agent_registration(self) -> AgentRegistration {
        let description = self.description().map(str::to_string);
        AgentRegistration {
            agent: Arc::new(self),
            description,
            instructions: None,
        }
    }
}

impl IntoAgentRegistration for Arc<WorkflowAgent> {
    fn into_agent_registration(self) -> AgentRegistration {
        let description = self.description().map(str::to_string);
        AgentRegistration {
            agent: self,
            description,
            instructions: None,
        }
    }
}

impl IntoAgentRegistration for Arc<dyn Agent> {
    fn into_agent_registration(self) -> AgentRegistration {
        AgentRegistration::new(self)
    }
}

/// A registered agent and its cached metadata.
pub(crate) struct AgentRecord {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) description: Option<String>,
    pub(crate) instructions: Option<String>,
    pub(crate) agent: Arc<dyn Agent>,
}

/// A registered workflow and its cached metadata.
pub(crate) struct WorkflowRecord {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) description: Option<String>,
    pub(crate) workflow: Workflow,
}

/// One registered entity: an agent or a workflow.
pub(crate) enum EntityRecord {
    Agent(AgentRecord),
    Workflow(WorkflowRecord),
}

/// Shared, immutable registry backing the router (held behind an `Arc`).
pub(crate) struct HostState {
    entities: Vec<EntityRecord>,
    index: HashMap<String, usize>,
}

impl HostState {
    /// All entities in registration order.
    pub(crate) fn list(&self) -> &[EntityRecord] {
        &self.entities
    }

    /// Look up an entity by its id.
    pub(crate) fn get(&self, id: &str) -> Option<&EntityRecord> {
        self.index.get(id).map(|&i| &self.entities[i])
    }
}

/// Builder and registry for agents and workflows served over HTTP.
///
/// ```no_run
/// # use std::sync::Arc;
/// # use agent_framework_core::agent::ChatAgent;
/// # use agent_framework_hosting::AgentHost;
/// # async fn demo(weather: ChatAgent) -> std::io::Result<()> {
/// let host = AgentHost::new().agent("weather", weather);
/// host.serve(([127, 0, 0, 1], 8080)).await
/// # }
/// ```
#[derive(Default)]
pub struct AgentHost {
    entities: Vec<EntityRecord>,
    index: HashMap<String, usize>,
}

impl AgentHost {
    /// A new, empty host.
    pub fn new() -> Self {
        Self::default()
    }

    fn insert(&mut self, id: String, record: EntityRecord) {
        if let Some(&existing) = self.index.get(&id) {
            // Replace an entity registered under the same id.
            self.entities[existing] = record;
        } else {
            self.index.insert(id, self.entities.len());
            self.entities.push(record);
        }
    }

    /// Register an agent under `name`, which becomes its entity id.
    ///
    /// Accepts a [`ChatAgent`], a [`WorkflowAgent`], or an `Arc<dyn Agent>`
    /// (see [`IntoAgentRegistration`]).
    pub fn agent(mut self, name: impl Into<String>, agent: impl IntoAgentRegistration) -> Self {
        let id = name.into();
        let reg = agent.into_agent_registration();
        let display = reg
            .agent
            .name()
            .map(str::to_string)
            .unwrap_or_else(|| id.clone());
        let record = EntityRecord::Agent(AgentRecord {
            id: id.clone(),
            name: display,
            description: reg.description,
            instructions: reg.instructions,
            agent: reg.agent,
        });
        self.insert(id, record);
        self
    }

    /// Register a workflow under `name`, which becomes its entity id.
    pub fn workflow(mut self, name: impl Into<String>, workflow: Workflow) -> Self {
        let id = name.into();
        let display = workflow
            .name()
            .map(str::to_string)
            .unwrap_or_else(|| id.clone());
        let description = workflow.description().map(str::to_string);
        let record = EntityRecord::Workflow(WorkflowRecord {
            id: id.clone(),
            name: display,
            description,
            workflow,
        });
        self.insert(id, record);
        self
    }

    /// The number of registered entities.
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// Whether no entities are registered.
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    pub(crate) fn into_state(self) -> Arc<HostState> {
        Arc::new(HostState {
            entities: self.entities,
            index: self.index,
        })
    }

    /// Build the DevUI-style [`axum::Router`] serving all registered entities.
    ///
    /// Routes: `GET /health`, `GET /v1/entities`, `GET /v1/entities/{id}/info`,
    /// `POST /v1/responses`. The router carries no state of its own beyond the
    /// registry and is freely nestable into a larger application.
    pub fn into_router(self) -> axum::Router {
        crate::devui::router(self.into_state())
    }

    /// Bind `addr` and serve the DevUI router until the process exits.
    pub async fn serve(self, addr: impl Into<std::net::SocketAddr>) -> std::io::Result<()> {
        let listener = tokio::net::TcpListener::bind(addr.into()).await?;
        axum::serve(listener, self.into_router()).await
    }
}
