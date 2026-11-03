//! Typed protocol descriptors: edge compatibility at build, workflow
//! protocol description and reflection.

use std::sync::Arc;

use agent_framework_core::workflow::{
    FunctionExecutor, ProtocolDescriptor, RequestInfoExecutor, WorkflowBuilder,
};

fn node(id: &str, protocol: ProtocolDescriptor) -> Arc<FunctionExecutor> {
    Arc::new(FunctionExecutor::new(id, |_m, _c| async move { Ok(()) }).with_protocol(protocol))
}

#[test]
fn an_edge_between_incompatible_declared_types_is_refused() {
    let build = |accepted: &str| {
        WorkflowBuilder::new()
            .add_executor(node(
                "a",
                ProtocolDescriptor::builder()
                    .accepts("Order")
                    .sends_message_type("Invoice")
                    .build(),
            ))
            .add_executor(node(
                "b",
                ProtocolDescriptor::builder()
                    .accepts(accepted)
                    .yields_output_type("Receipt")
                    .build(),
            ))
            .set_start("a")
            .add_edge("a", "b")
            .build()
    };
    let err = build("Payment").err().expect("refused");
    assert!(
        err.to_string().contains("TYPE_COMPATIBILITY") || err.to_string().contains("accepts"),
        "{err}"
    );
    let workflow = build("Invoice").unwrap();
    let protocol = workflow.describe_protocol();
    assert_eq!(protocol.accepts, ["Order"]);
    assert_eq!(protocol.yields, ["Receipt"]);
}

#[test]
fn undeclared_executors_accept_anything() {
    let workflow = WorkflowBuilder::new()
        .add_executor(node(
            "a",
            ProtocolDescriptor::builder()
                .sends_message_type("X")
                .build(),
        ))
        .add_executor(Arc::new(FunctionExecutor::new("b", |_m, _c| async move {
            Ok(())
        })))
        .set_start("a")
        .add_edge("a", "b")
        .build()
        .unwrap();
    assert_eq!(workflow.reflect_executors(), ["a", "b"]);
    assert_eq!(
        workflow.reflect_edges(),
        [("a".to_string(), "b".to_string())]
    );
}

#[test]
fn request_ports_are_reflected_with_their_types() {
    let workflow = WorkflowBuilder::new()
        .add_executor(node(
            "ask",
            ProtocolDescriptor::builder()
                .sends_message_type("Question")
                .build(),
        ))
        .add_executor(Arc::new(
            RequestInfoExecutor::new("human").with_types("Question", "Answer"),
        ))
        .set_start("ask")
        .add_edge("ask", "human")
        .build()
        .unwrap();
    let ports = workflow.reflect_ports();
    assert_eq!(ports.len(), 1);
    assert_eq!(ports[0].port_id, "human");
    assert_eq!(ports[0].request.as_deref(), Some("Question"));
    assert_eq!(ports[0].response.as_deref(), Some("Answer"));
}
