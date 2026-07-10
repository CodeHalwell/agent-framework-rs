# agent-framework-rs

A Rust implementation of the [Microsoft Agent Framework](https://github.com/microsoft/agent-framework)
for building AI agents and multi-agent workflows.

This project ports the framework's architecture — chat clients, agents, tools,
threads, memory, middleware, a graph-based workflow engine, a family of
prebuilt multi-agent orchestrations, and the surrounding ecosystem (A2A,
declarative specs, HTTP hosting, memory backends) — to idiomatic, async Rust,
with the goal of complete feature parity with the Python and .NET
implementations.

> Status: **near parity.** The core runtime; OpenAI (chat + Responses API),
> Azure OpenAI, and Anthropic providers; MCP tools (stdio / HTTP / WebSocket);
> the full workflow engine (checkpointing, human-in-the-loop, shared state,
> validation, visualization, sub-workflows); every prebuilt orchestration
> (sequential, concurrent, group chat, handoff, Magentic incl. plan review,
> workflow-as-agent); an A2A client; declarative YAML agents/workflows; HTTP
> hosting (DevUI-style, A2A, OpenAI-compatible); and Redis/Mem0 memory
> backends are implemented and tested. See [PARITY.md](PARITY.md) for the
> detailed, grounded feature-parity matrix and the [roadmap](#roadmap) below
> for the (short) list of what's left.

## Highlights

- **Agents** — a `ChatAgent` with instructions, tools, default options,
  conversation threads, memory/context providers, and middleware at all three
  levels: agent-run, chat-client-call, and per-tool-invocation.
- **Automatic tool calling** — the function-invocation loop executes local
  tools and feeds results back to the model until it produces a final answer.
- **Human-in-the-loop tool approval** — mark a tool `ApprovalMode::AlwaysRequire`
  and the loop pauses with a `FunctionApprovalRequestContent` instead of
  running it, until you resubmit an approval/rejection.
- **Structured output** — request a JSON-Schema-conforming response with
  `ResponseFormat::json_schema` and parse it with `response.parse_json::<T>()`.
  Works on every provider — for Anthropic (which has no native
  `response_format`) it's folded into the system prompt, which the Python and
  .NET references don't do at all.
- **Streaming** — token-by-token streaming for both chat clients and agents.
- **Workflows** — a Pregel-style, superstep graph engine (`WorkflowBuilder`)
  with single edges, conditions, fan-out, fan-in, and switch/case routing,
  plus checkpointing (in-memory and file-backed), human-in-the-loop
  request/response, run-scoped shared state, graph validation, Mermaid/DOT
  visualization, and sub-workflow composition.
- **Orchestration** — prebuilt `SequentialBuilder`, `ConcurrentBuilder`,
  `GroupChatBuilder` (round-robin, custom, or LLM-managed), `HandoffBuilder`,
  and `MagenticBuilder` (with optional human plan review before execution)
  patterns, plus `WorkflowAgent` to expose any workflow as an `Agent`.
- **Provider-agnostic core** — the `ChatClient` trait lets you plug in any
  backend; OpenAI (Chat Completions + Responses API), Azure OpenAI (API key
  or Microsoft Entra ID), and Anthropic providers ship in the box.
- **MCP** — connect to Model Context Protocol servers over stdio, streamable
  HTTP, or WebSocket and wire their tools straight into an agent.
- **A2A** — call any Agent2Agent-protocol server as a local `Agent`
  (`A2AAgent`), with agent-card discovery and multi-turn context continuity.
- **Declarative** — load agents from the official YAML spec vocabulary and
  workflows from a Rust-native spec, with env interpolation and pluggable
  provider/tool registries.
- **Hosting** — serve agents/workflows over HTTP with axum: a DevUI-style API
  (`/v1/entities`, `/v1/responses` with SSE), A2A serving (agent card +
  JSON-RPC), and an OpenAI-compatible `/v1/chat/completions`.
- **Memory backends** — Redis-backed conversation history and long-term
  memory, and a Mem0 REST provider, both as drop-in `ChatMessageStore` /
  `ContextProvider` implementations.
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
| [`agent-framework-mcp`](crates/agent-framework-mcp) | Model Context Protocol client (stdio, streamable HTTP, and WebSocket transports). |
| [`agent-framework-a2a`](crates/agent-framework-a2a) | Agent2Agent protocol client: `A2AAgent` + `A2AClient` (JSON-RPC, agent cards, streaming). |
| [`agent-framework-declarative`](crates/agent-framework-declarative) | Declarative YAML/JSON agents and workflows with provider/tool registries. |
| [`agent-framework-hosting`](crates/agent-framework-hosting) | Serve agents over HTTP (axum): DevUI-style API, A2A serving, OpenAI-compatible chat completions. |
| [`agent-framework-redis`](crates/agent-framework-redis) | Redis-backed `ChatMessageStore` and long-term-memory `ContextProvider`. |
| [`agent-framework-mem0`](crates/agent-framework-mem0) | Mem0 hosted-API long-term-memory `ContextProvider`. |
| [`agent-framework`](crates/agent-framework) | Umbrella crate re-exporting the core plus everything above behind cargo features. |

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
unconditionally, plus each companion crate behind a cargo feature:

| Feature | Crate | Default |
| --- | --- | --- |
| `openai` | [`agent-framework-openai`](crates/agent-framework-openai) — OpenAI Chat Completions + Responses API | yes |
| `anthropic` | [`agent-framework-anthropic`](crates/agent-framework-anthropic) — Anthropic (Claude) Messages API | no |
| `azure` | [`agent-framework-azure`](crates/agent-framework-azure) — Azure OpenAI (API-key / Entra ID) | no |
| `mcp` | [`agent-framework-mcp`](crates/agent-framework-mcp) — MCP tools (stdio, HTTP, WebSocket) | no |
| `a2a` | [`agent-framework-a2a`](crates/agent-framework-a2a) — Agent2Agent protocol client | no |
| `declarative` | [`agent-framework-declarative`](crates/agent-framework-declarative) — YAML/JSON agents & workflows | no |
| `hosting` | [`agent-framework-hosting`](crates/agent-framework-hosting) — serve agents over HTTP | no |
| `redis` | [`agent-framework-redis`](crates/agent-framework-redis) — Redis message store & memory | no |
| `mem0` | [`agent-framework-mem0`](crates/agent-framework-mem0) — Mem0 memory provider | no |
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
rest read their provider's credentials from environment variables (and skip
gracefully when unset).

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
| `hosting_server` | `AgentHost`: DevUI-style HTTP API with entity discovery + `/v1/responses` | `--features hosting` (offline with a canned agent; `OPENAI_API_KEY` optional) |
| `a2a_client` | `A2AAgent` against any A2A server, incl. multi-turn continuity | `A2A_AGENT_URL`, `--features a2a` |
| `declarative_agent` | Load a `ChatAgent` from an inline YAML spec via `DeclarativeLoader` | `--features declarative` (offline with a canned client; `OPENAI_API_KEY` optional) |
| `redis_memory` | Redis-backed thread history + long-term memory provider | a Redis server, `--features redis` (skips gracefully without one) |
| `mem0_memory` | Mem0 long-term memory provider | `MEM0_API_KEY` + `OPENAI_API_KEY`, `--features mem0` |
| `magentic_plan_review` | Magentic plan-review HITL: pause, revise, approve | offline — no key needed |
| `workflow_checkpoint` | `FileCheckpointStorage`: save, list, resume mid-pipeline | offline — no key needed |
| `workflow_hitl` | `RequestInfoExecutor` pause / `send_response` resume | offline — no key needed |
| `workflow_viz` | Render a branching workflow as Mermaid and Graphviz DOT | offline — no key needed |

```bash
OPENAI_API_KEY=sk-... cargo run -p agent-framework --example quickstart
cargo run -p agent-framework --example workflow_checkpoint      # offline
cargo run -p agent-framework --example magentic_plan_review     # offline
cargo run -p agent-framework --example hosting_server --features hosting
cargo run -p agent-framework --example declarative_agent --features declarative
ANTHROPIC_API_KEY=sk-ant-... cargo run -p agent-framework --example anthropic --features anthropic
```

## Design

The crate mirrors the Python framework's module structure so concepts map
one-to-one:

| Python (`agent_framework`) | Rust |
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
| `_mcp` | [`agent-framework-mcp`](crates/agent-framework-mcp) (`McpStdioTool`, `McpStreamableHttpTool`, `McpWebsocketTool`) |
| `a2a` package | [`agent-framework-a2a`](crates/agent-framework-a2a) (`A2AAgent`, `A2AClient`) |
| `declarative` package | [`agent-framework-declarative`](crates/agent-framework-declarative) (`DeclarativeLoader`) |
| `devui` package | [`agent-framework-hosting`](crates/agent-framework-hosting) (`AgentHost`, `A2ARouter`, `OpenAiRouter`) |
| `redis` / `mem0` packages | [`agent-framework-redis`](crates/agent-framework-redis) / [`agent-framework-mem0`](crates/agent-framework-mem0) |

Cross-cutting behavior implemented in Python via class decorators
(`use_function_invocation`, `use_*_middleware`) is expressed in Rust as wrapper
types (`FunctionInvokingChatClient`, `ObservableChatClient`) and explicit
middleware pipelines.

## Roadmap

See [PARITY.md](PARITY.md) for the full, grounded feature matrix. The
remaining gaps, roughly in the order they're likely to matter:

- [ ] A retry/backoff policy layer for chat clients
- [ ] MCP prompts (`prompts/list`/`prompts/get`) and sampling/roots callbacks
- [ ] A2A push notifications, `tasks/resubscribe`, authenticated extended card
- [ ] Checkpoint graph-signature validation on resume
- [ ] RediSearch-backed vector/hybrid retrieval for the Redis provider
      (currently recency + token match)
- [ ] The upstream Copilot-Studio declarative *workflow* DSL (declarative
      agents already follow the official schema; workflows use a documented
      Rust-native spec)
- [ ] A DevUI web frontend and stateful hosted runs (conversation store,
      run-resume endpoint)
- [ ] Azure AI (Foundry) service client and a real Entra ID credential chain
- [ ] OTel SDK exporter wiring (spans are emitted and bridge-ready today)
- [ ] Remaining ecosystem integrations: AG-UI, CopilotStudio, Purview,
      ChatKit, Azure AI Search, CosmosDB, DurableTask/Azure Functions hosting

Done:

- [x] Core data model, `ChatClient` + function-invocation loop, approval flow,
      structured output (all providers, incl. Anthropic via prompt injection)
- [x] `ChatAgent`, threads, memory/context providers, `as_tool`, and
      middleware at all three levels (agent / chat / function)
- [x] OpenAI (Chat Completions + Responses API), Azure OpenAI, Anthropic
- [x] MCP client: stdio, streamable HTTP, and WebSocket transports
- [x] Graph workflow engine: supersteps, edges, conditions, fan-out/in,
      switch, checkpointing, HITL, shared state, validation, viz,
      sub-workflows
- [x] Every orchestration: sequential, concurrent, group chat, handoff,
      Magentic (incl. plan-review HITL), workflow-as-agent
- [x] A2A client (`A2AAgent`) and A2A serving (agent card + JSON-RPC)
- [x] Declarative YAML agents (official schema) and Rust-native workflow specs
- [x] HTTP hosting: DevUI-style API (SSE), OpenAI-compatible chat completions
- [x] Redis chat-message store + context provider; Mem0 provider
- [x] `tracing`-based observability with OTel GenAI semantic conventions

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
