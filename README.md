# agent-framework-rs

A Rust implementation of the [Microsoft Agent Framework](https://github.com/microsoft/agent-framework)
for building AI agents and multi-agent workflows.

This project ports the framework's architecture — chat clients, agents, tools,
threads, memory, middleware, and a graph-based workflow engine — to idiomatic,
async Rust, with the goal of complete feature parity with the Python and .NET
implementations.

> Status: **early, actively building toward parity.** The core runtime, an
> OpenAI-compatible provider, and sequential/concurrent orchestration are
> implemented and tested. See the [roadmap](#roadmap) for what's next.

## Highlights

- **Agents** — a `ChatAgent` with instructions, tools, default options,
  conversation threads, memory/context providers, and middleware.
- **Automatic tool calling** — the function-invocation loop executes local
  tools and feeds results back to the model until it produces a final answer.
- **Streaming** — token-by-token streaming for both chat clients and agents.
- **Workflows** — a Pregel-style, superstep graph engine (`WorkflowBuilder`)
  with single edges, conditions, fan-out, fan-in, and switch/case routing.
- **Orchestration** — prebuilt `SequentialBuilder` and `ConcurrentBuilder`
  patterns that compose agents into multi-agent workflows.
- **Provider-agnostic core** — the `ChatClient` trait lets you plug in any
  backend; an OpenAI (and OpenAI-compatible) provider ships in the box.

## Workspace layout

| Crate | Description |
| --- | --- |
| [`agent-framework-core`](crates/agent-framework-core) | Core abstractions: types, chat client, agents, tools, threads, memory, middleware, workflows. |
| [`agent-framework-openai`](crates/agent-framework-openai) | OpenAI / OpenAI-compatible chat client (Azure OpenAI, Ollama, etc.). |
| [`agent-framework`](crates/agent-framework) | Umbrella crate re-exporting the core plus providers. |

## Quick start

```toml
[dependencies]
agent-framework = "0.1"
tokio = { version = "1", features = ["full"] }
```

```rust
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIClient::from_env("gpt-4o-mini")?; // reads OPENAI_API_KEY

    let agent = ChatAgent::builder(client)
        .name("assistant")
        .instructions("You are a helpful, concise assistant.")
        .build();

    let response = agent.run_once("What is the capital of France?").await?;
    println!("{}", response.text());
    Ok(())
}
```

### Tools

```rust
use agent_framework::prelude::*;
use serde_json::json;

let get_weather = AiFunction::new(
    "get_weather",
    "Get the current weather for a city.",
    json!({
        "type": "object",
        "properties": { "city": { "type": "string" } },
        "required": ["city"]
    }),
    |args| async move {
        let city = args["city"].as_str().unwrap_or("unknown");
        Ok(json!({ "city": city, "temperature_c": 21, "condition": "sunny" }))
    },
)
.into_definition();

let agent = ChatAgent::builder(client).tool(get_weather).build();
let reply = agent.run_once("What's the weather in Paris?").await?;
```

### Workflows

```rust
use std::sync::Arc;
use agent_framework::prelude::*;
use agent_framework::workflow::SequentialBuilder;

let workflow = SequentialBuilder::new()
    .participants(vec![writer as Arc<dyn Agent>, editor])
    .build()?;

let result = workflow.run("Write about Rust").await?;
let final_output = result.last_output();
```

See [`crates/agent-framework/examples`](crates/agent-framework/examples) for
runnable examples:

```bash
OPENAI_API_KEY=sk-... cargo run -p agent-framework --example quickstart
OPENAI_API_KEY=sk-... cargo run -p agent-framework --example tools
OPENAI_API_KEY=sk-... cargo run -p agent-framework --example streaming
OPENAI_API_KEY=sk-... cargo run -p agent-framework --example workflow_sequential
```

## Design

The crate mirrors the Python framework's module structure so concepts map
one-to-one:

| Python (`agent_framework`) | Rust (`agent_framework_core`) |
| --- | --- |
| `_types` | `types` (`ChatMessage`, `Content`, `ChatResponse`, `ChatOptions`, …) |
| `_clients` | `client` (`ChatClient`, `FunctionInvokingChatClient`) |
| `_agents` | `agent` (`Agent`, `ChatAgent`) |
| `_tools` | `tools` (`Tool`, `AiFunction`, hosted tools) |
| `_threads` | `threads` (`AgentThread`, `ChatMessageStore`) |
| `_memory` | `memory` (`ContextProvider`, `AggregateContextProvider`) |
| `_middleware` | `middleware` (agent / chat / function pipelines) |
| `_workflows` | `workflow` (`Workflow`, `WorkflowBuilder`, `Executor`, orchestration) |

Cross-cutting behavior implemented in Python via class decorators
(`use_function_invocation`, `use_*_middleware`) is expressed in Rust as wrapper
types (`FunctionInvokingChatClient`) and explicit middleware pipelines.

## Roadmap

Toward full feature parity with the upstream framework:

- [x] Core data model (messages, content union, responses, options, usage)
- [x] `ChatClient` trait + automatic function-invocation loop
- [x] `ChatAgent`, threads, memory/context providers, middleware
- [x] OpenAI / OpenAI-compatible provider (chat completions, streaming, tools)
- [x] Graph workflow engine (supersteps, edges, conditions, fan-out/in, switch)
- [x] Sequential & concurrent orchestration
- [ ] Additional orchestration: group chat, handoff, magentic
- [ ] Workflow checkpointing & human-in-the-loop request/response
- [ ] Structured output parsing helpers & JSON-schema derivation from types
- [ ] More providers: Azure AI, Anthropic, and others
- [ ] MCP client/server, A2A, declarative agents
- [ ] OpenTelemetry-based observability
- [ ] DevUI

## Development

```bash
cargo build            # build all crates
cargo test             # run the test suite (no network required)
cargo clippy --all-targets
cargo fmt --check
```

## License

MIT. This is an independent port and is not affiliated with or endorsed by
Microsoft.
