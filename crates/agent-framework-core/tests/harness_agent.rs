//! The assembled harness agent (`experimental-harness`).
//!
//! Ported from upstream `test_harness_agent.py`.
#![cfg(feature = "experimental-harness")]

use std::sync::{Arc, Mutex};

use agent_framework_core::agent::SupportsAgentRun;
use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::harness::*;
use agent_framework_core::prelude::*;
use async_trait::async_trait;

/// Answers "done" and records each request's instructions and tool names.
#[derive(Clone, Default)]
struct Recorder(Arc<Mutex<Vec<Call>>>);

/// One request: its instructions, tool names and `max_tokens`.
type Call = (Option<String>, Vec<String>, Option<u32>);

#[async_trait]
impl ChatClient for Recorder {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> agent_framework_core::error::Result<ChatResponse> {
        let system: Vec<String> = options
            .instructions
            .iter()
            .cloned()
            .chain(
                messages
                    .iter()
                    .filter(|m| m.role.as_str() == "system")
                    .map(Message::text),
            )
            .collect();
        self.0.lock().unwrap().push((
            (!system.is_empty()).then(|| system.join("\n")),
            options.tools.iter().map(|t| t.name.clone()).collect(),
            options.max_tokens,
        ));
        Ok(ChatResponse::from_text("done"))
    }

    async fn get_streaming_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> agent_framework_core::error::Result<ChatStream> {
        unimplemented!("not used")
    }
}

#[test]
fn token_limits_are_validated() {
    let client = Arc::new(Recorder::default());
    let err = create_harness_agent(client.clone())
        .disable_file_memory()
        .max_context_window_tokens(100)
        .max_output_tokens(100)
        .build()
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("less than max_context_window_tokens"),
        "{err}"
    );
    assert!(create_harness_agent(client)
        .disable_file_memory()
        .max_output_tokens(0)
        .build()
        .is_err());
}

#[tokio::test]
async fn the_default_assembly_wires_instructions_providers_and_options() {
    let client = Recorder::default();
    let files = Arc::new(InMemoryAgentFileStore::new());
    let harness = create_harness_agent(Arc::new(client.clone()))
        .name("helper")
        .agent_instructions("Focus on tests.")
        .file_memory_store(Arc::new(InMemoryAgentFileStore::new()))
        .file_access_store(files)
        .max_context_window_tokens(10_000)
        .max_output_tokens(1_000)
        .build()
        .unwrap();
    assert_eq!(SupportsAgentRun::name(&harness), Some("helper"));
    assert!(harness.todo_provider().is_some());
    assert!(harness.mode_provider().is_some());
    assert!(harness.file_access_provider().is_some());
    assert!(harness.background_agents_provider().is_none());

    let mut session = harness.create_session();
    let response = harness
        .run(vec![Message::user("hi")], Some(&mut session))
        .await
        .unwrap();
    assert_eq!(response.text(), "done");

    let calls = client.0.lock().unwrap().clone();
    let (instructions, tools, max_tokens) = &calls[0];
    let instructions = instructions.as_deref().unwrap();
    assert!(
        instructions.starts_with(DEFAULT_HARNESS_INSTRUCTIONS.trim_end()),
        "{instructions}"
    );
    assert!(instructions.contains("Focus on tests."));
    assert!(instructions.contains("## File Access"));
    assert_eq!(*max_tokens, Some(1_000));
    for expected in ["todos_add", "file_memory_write", "file_access_read"] {
        assert!(
            tools.iter().any(|t| t == expected),
            "{expected} missing from {tools:?}"
        );
    }
}

#[tokio::test]
async fn opt_outs_and_background_agents() {
    let client = Recorder::default();
    let worker: Arc<dyn SupportsAgentRun> =
        Arc::new(Agent::builder(Recorder::default()).name("worker").build());
    let harness = create_harness_agent(Arc::new(client.clone()))
        .harness_instructions("")
        .disable_todo()
        .disable_mode()
        .disable_file_memory()
        .disable_tool_auto_approval()
        .background_agents([worker])
        .build()
        .unwrap();
    assert!(harness.todo_provider().is_none());
    assert!(harness.background_agents_provider().is_some());
    let mut session = harness.create_session();
    harness
        .run(vec![Message::user("hi")], Some(&mut session))
        .await
        .unwrap();
    let calls = client.0.lock().unwrap().clone();
    let (instructions, tools, max_tokens) = &calls[0];
    assert!(instructions
        .as_deref()
        .unwrap()
        .starts_with("## Background Agents"));
    assert!(
        tools.iter().all(|t| t.starts_with("background_agents_")),
        "{tools:?}"
    );
    assert_eq!(*max_tokens, None);
}

#[tokio::test]
async fn a_loop_wraps_the_agent_when_asked() {
    let client = Recorder::default();
    let todo = TodoProvider::new();
    let harness = create_harness_agent(Arc::new(client.clone()))
        .disable_file_memory()
        .todo_provider(todo.clone())
        .loop_should_continue(|ctx: &LoopContext| ctx.iteration < 3)
        .build()
        .unwrap();
    let mut session = harness.create_session();
    harness
        .run(vec![Message::user("go")], Some(&mut session))
        .await
        .unwrap();
    assert_eq!(client.0.lock().unwrap().len(), 3);
}
