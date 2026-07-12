//! Load a `Workflow` from a declarative spec.
//!
//! Unlike the official YAML *agent* schema `declarative_agent.rs` loads, the
//! upstream declarative *workflow* schema is a Power Platform / Copilot
//! Studio imperative DSL that doesn't map onto this port's Pregel/BSP graph
//! engine -- so `agent-framework-declarative` instead defines a documented
//! Rust-native `WorkflowSpec` that drives the existing `WorkflowBuilder` and
//! orchestration builders directly. It keeps the top-level `kind: Workflow`
//! key but supports two spec bodies (see
//! `crates/agent-framework-declarative/src/workflow.rs` for the full shape):
//!
//! - **Orchestration shorthand** -- `type: sequential | concurrent |
//!   group_chat | handoff` plus a `participants` list of agent ids.
//! - **Explicit graph** -- `nodes` (each wrapping an agent), `edges`,
//!   `start`, and optionally `fanOut`/`fanIn`/`switch` groups.
//!
//! Both forms resolve participant/node agents by id from an `AgentRegistry`
//! you populate yourself -- here, by loading two more declarative *agent*
//! specs through the same `ChatClientFactory` registry `declarative_agent.rs`
//! uses.
//!
//! Runs offline (a canned client stands in for OPENAI_API_KEY).
//!
//! ```bash
//! cargo run -p agent-framework-examples --example declarative_workflow
//! ```

use std::sync::Arc;

use agent_framework::declarative::{AgentRegistry, ChatClientFactory, DeclarativeLoader};
use agent_framework::prelude::*;
use async_trait::async_trait;
use futures::StreamExt;

/// A canned model whose replies are tagged with `name`, so each participant's
/// contribution stays visible in a multi-agent run without any API key.
#[derive(Clone)]
struct CannedClient {
    name: &'static str,
}

#[async_trait]
impl ChatClient for CannedClient {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        let last = messages.last().map(Message::text).unwrap_or_default();
        Ok(ChatResponse::from_text(format!(
            "[{}] (canned) {last}",
            self.name
        )))
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let resp = self.get_response(messages, options).await?;
        let updates = resp.messages.into_iter().map(|m| {
            Ok(ChatResponseUpdate {
                contents: m.contents,
                role: Some(m.role),
                ..Default::default()
            })
        });
        Ok(futures::stream::iter(updates.collect::<Vec<_>>()).boxed())
    }
}

/// Two agent specs (official YAML vocabulary, same as `declarative_agent.rs`)
/// that become the `writer` and `editor` participants below. The `model.id`
/// is what the shared factory (in `main`) uses to tell them apart.
const WRITER_SPEC: &str = r#"
kind: Prompt
name: Writer
instructions: Draft a one-sentence description of the given topic.
model:
  id: writer-model
  provider: OpenAI
  apiType: Chat
"#;

const EDITOR_SPEC: &str = r#"
kind: Prompt
name: Editor
instructions: Tighten the previous draft to no more than twelve words.
model:
  id: editor-model
  provider: OpenAI
  apiType: Chat
"#;

/// Orchestration shorthand: `type: sequential` runs `writer` then `editor`,
/// each seeing (and extending) the same running conversation.
const SEQUENTIAL_SPEC: &str = r#"
kind: Workflow
name: pipeline
type: sequential
participants:
  - writer
  - editor
"#;

/// The explicit node/edge graph form of the *same* two-stage pipeline: an
/// unconditional edge from `draft` to `polish`, with `polish` marked as the
/// workflow's output node. Graphs can also express conditional edges,
/// `fanOut`/`fanIn` groups, and `switch` branches -- see the spec docs.
const GRAPH_SPEC: &str = r#"
kind: Workflow
name: pipeline-graph
start: draft
nodes:
  - id: draft
    agent: writer
  - id: polish
    agent: editor
    output: true
edges:
  - from: draft
    to: polish
"#;

#[tokio::main]
async fn main() -> Result<()> {
    // One client factory, shared by every agent spec loaded below -- the
    // `model.id` in each spec (see WRITER_SPEC/EDITOR_SPEC) picks which
    // canned persona answers, mimicking how a real deployment would route
    // different model ids to different deployments through the same factory.
    let factory = ChatClientFactory::new().with("OpenAI.Chat", |model| {
        let name = match model.id.as_deref() {
            Some("editor-model") => "editor",
            _ => "writer",
        };
        Ok(Arc::new(CannedClient { name }) as Arc<dyn ChatClient>)
    });
    let loader = DeclarativeLoader::new().with_client_factory(factory);

    let writer = loader
        .load_agent(WRITER_SPEC)
        .map_err(|e| Error::Configuration(e.to_string()))?;
    let editor = loader
        .load_agent(EDITOR_SPEC)
        .map_err(|e| Error::Configuration(e.to_string()))?;

    // `load_workflow` resolves `participants` (shorthand) / `nodes[].agent`
    // (graph) references against this registry.
    let mut agents = AgentRegistry::new();
    agents.register_chat_agent("writer", writer);
    agents.register_chat_agent("editor", editor);

    println!("-- orchestration shorthand (type: sequential) --");
    let sequential = loader
        .load_workflow(SEQUENTIAL_SPEC, &agents)
        .map_err(|e| Error::Configuration(e.to_string()))?;
    let run = sequential.run("Rust's ownership model").await?;
    print_conversation(run.last_output());

    println!("\n-- explicit node/edge graph (same pipeline) --");
    let graph = loader
        .load_workflow(GRAPH_SPEC, &agents)
        .map_err(|e| Error::Configuration(e.to_string()))?;
    let run = graph.run("Rust's ownership model").await?;
    print_conversation(run.last_output());

    Ok(())
}

/// A workflow's output `Value` for an agent-backed node is the running
/// conversation, serialized as a JSON array of `Message`s (see
/// `AgentExecutor::execute`). Deserialize it back and print each turn.
fn print_conversation(output: Option<serde_json::Value>) {
    let Some(value) = output else {
        println!("(no output)");
        return;
    };
    match serde_json::from_value::<Vec<Message>>(value.clone()) {
        Ok(messages) => {
            for m in &messages {
                println!("  [{}] {}", m.role, m.text());
            }
        }
        Err(_) => println!("  {value}"),
    }
}
