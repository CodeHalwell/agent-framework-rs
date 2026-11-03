//! Typed protocol descriptors for executors and workflows.
//!
//! Port of upstream's `ProtocolDescriptor` / `ProtocolBuilder` /
//! `DescribeProtocol`. Messages in this port are JSON values, so a "type" is
//! a name: one the workflow's author chooses (`"Order"`, `"Approval"`) or a
//! JSON type name (`"string"`, `"object"`). An executor declares what it
//! accepts, sends and yields with [`Executor::describe_protocol`](super::Executor::describe_protocol);
//! [`WorkflowBuilder::build`](super::WorkflowBuilder::build) then refuses an
//! edge whose source sends nothing its target accepts, and
//! [`Workflow::describe_protocol`](super::Workflow::describe_protocol)
//! reports what the workflow as a whole accepts and yields.
//!
//! Executors that declare nothing accept everything, as before, so the check
//! only binds where both ends of an edge are declared.

use serde::{Deserialize, Serialize};

/// What an executor (or a workflow) accepts, sends and yields.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtocolDescriptor {
    /// The message types accepted; ignored when [`Self::accepts_all`].
    pub accepts: Vec<String>,
    /// Whether any message is accepted.
    pub accepts_all: bool,
    /// The message types sent downstream.
    pub sends: Vec<String>,
    /// The output types yielded.
    pub yields: Vec<String>,
}

impl ProtocolDescriptor {
    /// A descriptor accepting anything and declaring nothing else: the
    /// default for an executor that does not describe itself.
    pub fn accepts_all() -> Self {
        Self {
            accepts_all: true,
            ..Default::default()
        }
    }

    /// Start building a descriptor (upstream `ProtocolBuilder`).
    pub fn builder() -> ProtocolBuilder {
        ProtocolBuilder::default()
    }

    /// Whether a message of type `name` is accepted.
    pub fn accepts_type(&self, name: &str) -> bool {
        self.accepts_all || self.accepts.iter().any(|a| a == name)
    }
}

/// Builds a [`ProtocolDescriptor`].
#[derive(Debug, Clone, Default)]
pub struct ProtocolBuilder {
    descriptor: ProtocolDescriptor,
}

impl ProtocolBuilder {
    /// Accept messages of type `name`.
    pub fn accepts(mut self, name: impl Into<String>) -> Self {
        self.descriptor.accepts.push(name.into());
        self
    }

    /// Accept any message.
    pub fn accepts_all(mut self) -> Self {
        self.descriptor.accepts_all = true;
        self
    }

    /// Declare that messages of type `name` are sent (upstream
    /// `SendsMessageType`).
    pub fn sends_message_type(mut self, name: impl Into<String>) -> Self {
        self.descriptor.sends.push(name.into());
        self
    }

    /// Declare that outputs of type `name` are yielded (upstream
    /// `YieldsOutputType`).
    pub fn yields_output_type(mut self, name: impl Into<String>) -> Self {
        self.descriptor.yields.push(name.into());
        self
    }

    /// The descriptor.
    pub fn build(self) -> ProtocolDescriptor {
        self.descriptor
    }
}

/// A request port's request and response types (upstream `RequestPort`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestPortInfo {
    /// The port's executor id.
    pub port_id: String,
    /// The request type name, if declared.
    pub request: Option<String>,
    /// The response type name, if declared.
    pub response: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_builder_assembles_a_descriptor() {
        let d = ProtocolDescriptor::builder()
            .accepts("Order")
            .sends_message_type("Invoice")
            .yields_output_type("Receipt")
            .build();
        assert!(d.accepts_type("Order"));
        assert!(!d.accepts_type("Invoice"));
        assert_eq!(d.sends, ["Invoice"]);
        assert_eq!(d.yields, ["Receipt"]);
        assert!(ProtocolDescriptor::accepts_all().accepts_type("anything"));
    }
}
