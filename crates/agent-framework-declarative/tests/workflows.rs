//! Workflow-spec building and offline execution with mock-backed agents.

mod common;

use agent_framework_core::prelude::*;
use agent_framework_declarative::{
    AgentRegistry, DeclarativeError, DeclarativeLoader, WorkflowSpec,
};
use common::mock_agent;

fn registry() -> AgentRegistry {
    let mut agents = AgentRegistry::new();
    agents.register_chat_agent("writer", mock_agent("writer", "reply-A"));
    agents.register_chat_agent("editor", mock_agent("editor", "reply-B"));
    agents
}

/// `Workflow` does not implement `Debug`, so `Result::unwrap_err` is
/// unavailable; extract the error explicitly.
fn expect_err(loader: &DeclarativeLoader, yaml: &str, agents: &AgentRegistry) -> DeclarativeError {
    match loader.load_workflow(yaml, agents) {
        Ok(_) => panic!("expected load_workflow to fail:\n{yaml}"),
        Err(err) => err,
    }
}

#[tokio::test]
async fn sequential_shorthand_runs_pipeline() {
    let yaml = "\
kind: Workflow
name: pipeline
type: sequential
participants:
  - writer
  - editor
";
    let loader = DeclarativeLoader::new();
    let workflow = loader.load_workflow(yaml, &registry()).expect("build");
    assert_eq!(workflow.name(), Some("pipeline"));

    let run = workflow.run("draft this").await.expect("run");
    let output = run.last_output().expect("output");
    let text = output.to_string();
    assert!(text.contains("reply-A"), "missing writer reply: {text}");
    assert!(text.contains("reply-B"), "missing editor reply: {text}");
}

#[tokio::test]
async fn concurrent_shorthand_fans_out_and_in() {
    let yaml = "\
kind: Workflow
type: concurrent
participants: [writer, editor]
";
    let loader = DeclarativeLoader::new();
    let workflow = loader.load_workflow(yaml, &registry()).expect("build");
    let run = workflow.run("analyze this").await.expect("run");
    let text = run.last_output().expect("output").to_string();
    assert!(text.contains("reply-A"), "missing writer: {text}");
    assert!(text.contains("reply-B"), "missing editor: {text}");
}

#[tokio::test]
async fn explicit_graph_chain_runs() {
    let yaml = "\
kind: Workflow
name: graph
start: n1
nodes:
  - id: n1
    agent: writer
  - id: n2
    agent: editor
    output: true
edges:
  - from: n1
    to: n2
";
    let loader = DeclarativeLoader::new();
    let workflow = loader.load_workflow(yaml, &registry()).expect("build");
    let run = workflow.run("go").await.expect("run");
    assert_eq!(run.state(), WorkflowRunState::Idle);
    let text = run.last_output().expect("output").to_string();
    assert!(
        text.contains("reply-A") && text.contains("reply-B"),
        "{text}"
    );
}

#[tokio::test]
async fn group_chat_round_robin_builds_and_runs() {
    let yaml = "\
kind: Workflow
name: chat
type: group_chat
participants: [writer, editor]
roundRobin: true
maxRounds: 2
";
    let loader = DeclarativeLoader::new();
    let workflow = loader.load_workflow(yaml, &registry()).expect("build");
    assert_eq!(workflow.name(), Some("chat"));
    let run = workflow.run("kick off").await.expect("run");
    assert!(!run.events().is_empty());
}

#[test]
fn handoff_shorthand_builds() {
    let yaml = "\
kind: Workflow
type: handoff
participants: [writer, editor]
start: writer
handoffs:
  - from: writer
    to: [editor]
autonomous: true
";
    let loader = DeclarativeLoader::new();
    // Building exercises the HandoffBuilder wiring (initial agent + edges).
    loader
        .load_workflow(yaml, &registry())
        .expect("build handoff");
}

#[test]
fn unknown_agent_reference_is_actionable() {
    let yaml = "kind: Workflow\ntype: sequential\nparticipants: [ghost]\n";
    let loader = DeclarativeLoader::new();
    let err = expect_err(&loader, yaml, &registry());
    match &err {
        DeclarativeError::UnknownReference { kind, name, .. } => {
            assert_eq!(*kind, "agent");
            assert_eq!(name, "ghost");
        }
        other => panic!("expected UnknownReference, got {other:?}"),
    }
}

#[test]
fn unknown_predicate_reference_is_actionable() {
    let yaml = "\
kind: Workflow
start: n1
nodes:
  - id: n1
    agent: writer
  - id: n2
    agent: editor
    output: true
edges:
  - from: n1
    to: n2
    predicate: never_registered
";
    let loader = DeclarativeLoader::new();
    let err = expect_err(&loader, yaml, &registry());
    assert!(
        matches!(&err, DeclarativeError::UnknownReference { kind, .. } if *kind == "predicate"),
        "got {err:?}"
    );
}

#[test]
fn empty_workflow_is_actionable() {
    let yaml = "kind: Workflow\nname: empty\n";
    let loader = DeclarativeLoader::new();
    let err = expect_err(&loader, yaml, &registry());
    assert!(matches!(err, DeclarativeError::Invalid(_)), "got {err:?}");
}

#[tokio::test]
async fn predicate_routes_explicit_graph() {
    // A predicate that inspects the running conversation (a JSON array of
    // messages) and matches when the first message is from the user.
    let mut predicates = agent_framework_declarative::PredicateRegistry::new();
    predicates.register("first_is_user", |value: &serde_json::Value| {
        value
            .as_array()
            .and_then(|a| a.first())
            .and_then(|m| m.get("role"))
            .and_then(|r| r.as_str())
            == Some("user")
    });

    let yaml = "\
kind: Workflow
start: n1
nodes:
  - id: n1
    agent: writer
  - id: n2
    agent: editor
    output: true
edges:
  - from: n1
    to: n2
    predicate: first_is_user
";
    let loader = DeclarativeLoader::new().with_predicates(predicates);
    let workflow = loader.load_workflow(yaml, &registry()).expect("build");
    let run = workflow.run("hello").await.expect("run");
    // The first message stays the user turn, so the predicate holds and n2 runs.
    let text = run.last_output().expect("output").to_string();
    assert!(text.contains("reply-B"), "editor should have run: {text}");
}

#[test]
fn workflow_spec_round_trips() {
    let yaml = "\
kind: Workflow
name: graph
start: n1
maxIterations: 25
nodes:
  - id: n1
    agent: writer
  - id: n2
    agent: editor
    output: true
edges:
  - from: n1
    to: n2
    condition: status == \"ready\"
switch:
  - from: n1
    cases:
      - condition: n >= 3
        to: n2
        label: big
    default: n2
";
    let spec = WorkflowSpec::from_yaml(yaml).expect("parse");
    let reparsed = WorkflowSpec::from_yaml(&spec.to_yaml().expect("serialize")).expect("reparse");
    assert_eq!(spec, reparsed);
}
