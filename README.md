# agent-framework-rs

A Rust implementation of the [Microsoft Agent Framework](https://github.com/microsoft/agent-framework)
for building AI agents and multi-agent workflows.

This project ports the framework's architecture — chat clients, agents, tools,
threads, memory, middleware, a graph-based workflow engine, and a family of
prebuilt multi-agent orchestrations — to idiomatic, async Rust, with the goal
of complete feature parity with the Python and .NET implementations.

> Status: **actively building toward parity.** The core runtime; OpenAI
> (chat + Responses API), Azure OpenAI, and Anthropic providers; MCP tools;
> the full workflow engine (checkpointing, human-in-the-loop, shared state,
> validation, visualization, sub-workflows); and every prebuilt orchestration
> (sequential, concurrent, group chat, handoff, Magentic, workflow-as-agent)
> are implemented and tested. See [PARITY.md](PARITY.md) for a detailed,
> grounded feature-parity matrix against Python and .NET, and the
> [roadmap](#roadmap) below for what's next.

## Highlights

- **Agents** — a `ChatAgent` with instructions, tools, default options,
  conversation threads, memory/context providers, and middleware.
- **Automatic tool calling** — the function-invocation loop executes local
  tools and feeds results back to the model until it produces a final answer.
- **Human-in-the-loop tool approval** — mark a tool `ApprovalMode::AlwaysRequire`
  and the loop pauses with a `FunctionApprovalRequestContent` instead of
  running it, until you resubmit an approval/rejection.
- **Structured output** — request a JSON-Schema-conforming response with
  `ResponseFormat::json_schema` and parse it with `response.parse_json::<T>()`.
- **Streaming** — token-by-token streaming for both chat clients and agents.
- **Workflows** — a Pregel-style, superstep graph engine (`WorkflowBuilder`)
  with single edges, conditions, fan-out, fan-in, and switch/case routing,
  plus checkpointing (in-memory and file-backed), human-in-the-loop
  request/response, run-scoped shared state, graph validation, Mermaid/DOT
  visualization, and sub-workflow composition.
- **Orchestration** — prebuilt `SequentialBuilder`, `ConcurrentBuilder`,
  `GroupChatBuilder` (round-robin, custom, or LLM-managed), `HandoffBuilder`,
  and `MagenticBuilder` patterns that compose agents into multi-agent
  workflows, plus `WorkflowAgent` to expose any workflow as an `Agent`
  (including as a tool for another agent).
- **Provider-agnostic core** — the `ChatClient` trait lets you plug in any
  backend; OpenAI (Chat Completions + Responses API), Azure OpenAI (API key
  or Microsoft Entra ID), and Anthropic providers ship in the box.
- **MCP** — connect to Model Context Protocol servers over stdio or
  streamable HTTP and wire their tools straight into an agent.
- **Observability** — `ObservableChatClient` emits `tracing` spans following
  OpenTelemetry GenAI semantic conventions, ready to bridge into any OTel
  exporter.

## Workspace layout

| Crate | Description |
| --- | --- |
| [`agent-framework-core`](crates/agent-framework-core) | Core abstractions: types, chat client, agents, tools, threads, memory, middleware, observability, workflows & orchestration. |
| [`agent-framework-openai`](crates/agent-framework-openai) | OpenAI Chat Completions and Responses API clients (also used for OpenAI-compatible endpoints, e.g. Ollama). |
| [`agent-framework-anthropic`](crates/agent-framework-anthropic) | Anthropic (Claude) Messages API client. |
| [`agent-framework-azure`](crates/agent-framework-azure) | Azure OpenAI client (API-key and Microsoft Entra ID authentication). |
| [`agent-framework-mcp`](crates/agent-framework-mcp) | Model Context Protocol client (stdio and streamable HTTP transports). |
| [`agent-framework`](crates/agent-framework) | Umbrella crate re-exporting the core plus every provider behind cargo features. |

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

## Feature flags

The umbrella `agent-framework` crate re-exports [`agent-framework-core`]
unconditionally, plus each provider behind a cargo feature:

| Feature | Crate | Default |
| --- | --- | --- |
| `openai` | [`agent-framework-openai`](crates/agent-framework-openai) — OpenAI Chat Completions + Responses API | yes |
| `anthropic` | [`agent-framework-anthropic`](crates/agent-framework-anthropic) — Anthropic (Claude) Messages API | no |
| `azure` | [`agent-framework-azure`](crates/agent-framework-azure) — Azure OpenAI (API-key / Entra ID) | no |
| `mcp` | [`agent-framework-mcp`](crates/agent-framework-mcp) — Model Context Protocol tools | no |
| `full` | all of the above | no |

```toml
# Everything:
agent-framework = { version = "0.1", features = ["full"] }
# Just OpenAI (the default) plus Anthropic:
agent-framework = { version = "0.1", features = ["anthropic"] }
```

## Examples

See [`crates/agent-framework/examples`](crates/agent-framework/examples) for
runnable examples. The offline ones need no API key or network access; the
rest read their provider's API key from an environment variable and talk to
the real service.

| Example | Shows | Requires |
| --- | --- | --- |
| `quickstart` | Minimal `ChatAgent` + OpenAI | `OPENAI_API_KEY` |
| `streaming` | Token-by-token agent streaming | `OPENAI_API_KEY` |
| `tools` | Local tool calling via the function-invocation loop | `OPENAI_API_KEY` |
| `structured_output` | `ResponseFormat::json_schema` + `response.parse_json::<T>()` | `OPENAI_API_KEY` |
| `approvals` | Human-in-the-loop tool approval loop | `OPENAI_API_KEY` |
| `agent_as_tool` | A specialist agent exposed via `.as_tool()` to an orchestrator | `OPENAI_API_KEY` |
| `observability` | `ObservableChatClient` + `tracing_subscriber` span output | `OPENAI_API_KEY` |
| `openai_responses` | OpenAI Responses API + `conversation_id` reuse | `OPENAI_API_KEY` |
| `workflow_sequential` | Two-agent sequential pipeline (`SequentialBuilder`) | `OPENAI_API_KEY` |
| `group_chat` | Multi-agent group chat: round-robin and LLM-managed variants | `OPENAI_API_KEY` |
| `handoff` | Triage agent handing off to specialists | `OPENAI_API_KEY` |
| `magentic` | `MagenticBuilder` + `StandardMagenticManager` over two participants | `OPENAI_API_KEY` |
| `anthropic` | Anthropic Messages API | `ANTHROPIC_API_KEY`, `--features anthropic` |
| `azure_openai` | Azure OpenAI, API-key and Entra ID auth | `AZURE_OPENAI_*` vars, `--features azure` |
| `mcp_tools` | Connect an MCP stdio server, wire its tools into an agent | `OPENAI_API_KEY`, Node/`npx`, `--features mcp` |
| `workflow_checkpoint` | `FileCheckpointStorage`: save, list, resume mid-pipeline | offline — no key needed |
| `workflow_hitl` | `RequestInfoExecutor` pause / `send_response` resume | offline — no key needed |
| `workflow_viz` | Render a branching workflow as Mermaid and Graphviz DOT | offline — no key needed |

```bash
OPENAI_API_KEY=sk-... cargo run -p agent-framework --example quickstart
OPENAI_API_KEY=sk-... cargo run -p agent-framework --example group_chat
cargo run -p agent-framework --example workflow_checkpoint  # offline

ANTHROPIC_API_KEY=sk-ant-... cargo run -p agent-framework --example anthropic --features anthropic
cargo run -p agent-framework --example mcp_tools --features mcp
```

## Design

The crate mirrors the Python framework's module structure so concepts map
one-to-one:

| Python (`agent_framework`) | Rust (`agent_framework_core`) |
| --- | --- |
| `_types` | `types` (`ChatMessage`, `Content`, `ChatResponse`, `ChatOptions`, `ResponseFormat`, …) |
| `_clients` | `client` (`ChatClient`, `FunctionInvokingChatClient`) |
| `_agents` | `agent` (`Agent`, `ChatAgent`, `as_tool`) |
| `_tools` | `tools` (`Tool`, `AiFunction`, hosted tools, `ApprovalMode`) |
| `_threads` | `threads` (`AgentThread`, `ChatMessageStore`) |
| `_memory` | `memory` (`ContextProvider`, `AggregateContextProvider`) |
| `_middleware` | `middleware` (agent / chat / function pipelines) |
| `observability` | `observability` (`ObservableChatClient`, OTel GenAI spans) |
| `_workflows` | `workflow` (`Workflow`, `WorkflowBuilder`, `Executor`, checkpointing, HITL, shared state, validation, viz, sub-workflows) |
| `_workflows._sequential` / `_concurrent` / `_group_chat` / `_handoff` / `_magentic` / `_agent` | `workflow::orchestration` (`SequentialBuilder`, `ConcurrentBuilder`, `GroupChatBuilder`, `HandoffBuilder`, `MagenticBuilder`, `WorkflowAgent`) |
| `_mcp` | [`agent-framework-mcp`](crates/agent-framework-mcp) (`McpStdioTool`, `McpStreamableHttpTool`) |

Cross-cutting behavior implemented in Python via class decorators
(`use_function_invocation`, `use_*_middleware`) is expressed in Rust as wrapper
types (`FunctionInvokingChatClient`, `ObservableChatClient`) and explicit
middleware pipelines.

## Roadmap

See [PARITY.md](PARITY.md) for the full, grounded feature matrix. What's
genuinely still missing, roughly in the order it's likely to matter:

- [ ] Anthropic structured-output mapping (`ResponseFormat` is currently a
      no-op against `AnthropicClient`)
- [ ] A retry/backoff policy layer for chat clients
- [ ] MCP: WebSocket transport, prompts (`prompts/list`/`prompts/get`),
      sampling/roots callbacks
- [ ] Magentic human-in-the-loop plan review
- [ ] `WorkflowAgent` updating the caller's thread history after a run
- [ ] Checkpoint graph-signature validation on resume
- [ ] Chat- and function-invocation middleware actually wired into their
      call paths (the types exist; only agent middleware is invoked today)
- [ ] Mem0- and Redis-backed memory providers
- [ ] A2A, AG-UI, DevUI, hosting integrations, declarative/YAML agent and
      workflow definitions, CopilotStudio

Done:

- [x] Core data model (messages, content union, responses, options, usage,
      structured-output request/parse)
- [x] `ChatClient` trait + automatic function-invocation loop + approval flow
- [x] `ChatAgent`, threads, memory/context providers, middleware, `as_tool`
- [x] OpenAI (Chat Completions + Responses API), Azure OpenAI, and Anthropic
      providers
- [x] MCP client (stdio + streamable HTTP transports)
- [x] Graph workflow engine: supersteps, edges, conditions, fan-out/in,
      switch, checkpointing, human-in-the-loop, shared state, validation,
      Mermaid/DOT visualization, sub-workflows
- [x] Every prebuilt orchestration: sequential, concurrent, group chat
      (round-robin / custom / LLM-managed), handoff, Magentic,
      workflow-as-agent
- [x] `tracing`-based observability with OpenTelemetry GenAI semantic
      conventions

## Development

```bash
cargo build            # build all crates
cargo test              # run the test suite (no network required)
cargo clippy --all-targets
cargo fmt --check
```

## License

MIT. This is an independent port and is not affiliated with or endorsed by
Microsoft.
