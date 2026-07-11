# agent-framework-rs

A Rust implementation of the [Microsoft Agent Framework](https://github.com/microsoft/agent-framework)
for building AI agents and multi-agent workflows.

This project ports the framework's architecture — chat clients, agents, tools,
threads, memory, middleware, a graph-based workflow engine, a family of
prebuilt multi-agent orchestrations, and the surrounding ecosystem (A2A,
AG-UI, declarative specs, HTTP hosting, memory backends, compliance
middleware) — to idiomatic, async Rust, with the goal of complete feature
parity with the Python and .NET implementations.

> Status: **at parity for the core and nearly all of the ecosystem.** The
> core runtime with retries and three middleware levels; OpenAI (chat +
> Responses API), Azure OpenAI, Azure AI Foundry, Anthropic, and Copilot
> Studio providers; a full Entra ID credential chain; MCP tools (stdio /
> HTTP / WebSocket, with prompts, sampling, and roots); the workflow engine
> (signature-validated checkpointing, human-in-the-loop, shared state,
> validation, visualization, sub-workflows); every orchestration (sequential,
> concurrent, group chat, handoff, Magentic with plan-review and
> stall-intervention HITL, workflow-as-agent); a complete A2A client; YAML
> agents/workflows; HTTP hosting (DevUI-style + embedded debug page, A2A,
> AG-UI, OpenAI-compatible); Redis, Mem0, Cosmos DB, and Azure AI Search
> memory/storage; and Purview compliance middleware are implemented and
> tested. See [PARITY.md](PARITY.md) for the detailed, grounded matrix and
> the [roadmap](#roadmap) for the short list of what's left.

## Highlights

- **Agents** — a `ChatAgent` with instructions, tools, default options,
  conversation threads, memory/context providers, and middleware at all three
  levels: agent-run, chat-client-call, and per-tool-invocation.
- **Automatic tool calling** — the function-invocation loop executes local
  tools and feeds results back to the model until it produces a final answer.
- **Retries** — `RetryingChatClient` wraps any client with a `RetryPolicy`:
  exponential backoff with jitter, retryable-status classification, and
  server `Retry-After` honored over the computed delay.
- **Human-in-the-loop everywhere** — tool-call approval
  (`ApprovalMode::AlwaysRequire`), workflow `request_info` pauses, Magentic
  plan review before execution, and Magentic stall intervention mid-run.
- **Structured output** — request a JSON-Schema-conforming response with
  `ResponseFormat::json_schema` and parse it with `response.parse_json::<T>()`.
  Works on every provider — for Anthropic (no native `response_format`) it's
  folded into the system prompt, which the Python and .NET references don't
  do at all.
- **Streaming** — token-by-token streaming for both chat clients and agents.
- **Workflows** — a Pregel-style, superstep graph engine (`WorkflowBuilder`)
  with single edges, conditions, fan-out, fan-in, and switch/case routing,
  plus checkpointing (in-memory and file-backed, with graph-signature
  validation on resume), human-in-the-loop request/response, run-scoped
  shared state, graph validation, Mermaid/DOT visualization, and sub-workflow
  composition.
- **Orchestration** — prebuilt `SequentialBuilder`, `ConcurrentBuilder`,
  `GroupChatBuilder` (round-robin, custom, or LLM-managed), `HandoffBuilder`,
  and `MagenticBuilder` patterns, plus `WorkflowAgent` to expose any workflow
  as an `Agent`.
- **Providers** — OpenAI (Chat Completions + Responses API), Azure OpenAI
  (API key or Entra ID), Azure AI Foundry persistent agents, Anthropic, and
  Copilot Studio (Direct-to-Engine), all behind the `ChatClient`/`Agent`
  abstractions; plus a hand-rolled Entra ID credential chain
  (`AzureCliCredential`, `ClientSecretCredential`, `ManagedIdentityCredential`,
  `ChainedTokenCredential`).
- **MCP** — stdio, streamable HTTP, and WebSocket transports; tools, prompts,
  server-initiated sampling (`chat_client_sampling_handler` answers
  `sampling/createMessage` with your model), and roots.
- **A2A** — call any Agent2Agent server as a local `Agent` (`A2AAgent`), with
  card discovery, multi-turn context continuity, streaming, push-notification
  config, task resubscription, and the authenticated extended card.
- **Declarative** — load agents from the official YAML spec vocabulary and
  workflows from a Rust-native spec, with env interpolation and pluggable
  provider/tool registries.
- **Hosting** — serve agents/workflows over HTTP with axum: a DevUI-style API
  (`/v1/entities`, `/v1/responses` with SSE) with an embedded zero-dependency
  debug page at `/`, A2A serving (agent card + JSON-RPC), the AG-UI protocol
  (`AgUiRouter` SSE events for CopilotKit frontends), and an OpenAI-compatible
  `/v1/chat/completions`.
- **Memory & storage** — Redis-backed history and long-term memory (BM25 via
  RediSearch, SCAN fallback on plain Redis), Mem0, Azure Cosmos DB message
  store, and an Azure AI Search context provider.
- **Compliance** — Purview middleware evaluates prompts and responses against
  Microsoft Graph `processContent` and blocks on DLP verdicts.
- **Observability** — `ObservableChatClient` emits `tracing` spans following
  OpenTelemetry GenAI semantic conventions, ready to bridge into any OTel
  exporter.

## Workspace layout

| Crate | Description |
| --- | --- |
| [`agent-framework-core`](crates/agent-framework-core) | Core abstractions: types, chat client + retries, agents, tools, threads, memory, middleware, observability, workflows & orchestration. |
| [`agent-framework-openai`](crates/agent-framework-openai) | OpenAI Chat Completions and Responses API clients (also for OpenAI-compatible endpoints). |
| [`agent-framework-anthropic`](crates/agent-framework-anthropic) | Anthropic (Claude) Messages API client. |
| [`agent-framework-azure`](crates/agent-framework-azure) | Azure OpenAI client + the Entra ID credential chain (CLI / client-secret / managed-identity / chained). |
| [`agent-framework-azure-ai`](crates/agent-framework-azure-ai) | Azure AI Foundry persistent-agents client (agents, threads, runs, SSE). |
| [`agent-framework-azure-ai-search`](crates/agent-framework-azure-ai-search) | Azure AI Search context provider (semantic + optional vector query). |
| [`agent-framework-mcp`](crates/agent-framework-mcp) | MCP client: stdio/HTTP/WebSocket transports, tools, prompts, sampling, roots. |
| [`agent-framework-a2a`](crates/agent-framework-a2a) | Agent2Agent protocol client: `A2AAgent` + `A2AClient` (full task surface). |
| [`agent-framework-declarative`](crates/agent-framework-declarative) | Declarative YAML/JSON agents and workflows with provider/tool registries. |
| [`agent-framework-hosting`](crates/agent-framework-hosting) | HTTP serving (axum): DevUI-style API + embedded debug UI, A2A, AG-UI, OpenAI-compatible. |
| [`agent-framework-redis`](crates/agent-framework-redis) | Redis-backed `ChatMessageStore` and long-term-memory `ContextProvider` (RediSearch BM25). |
| [`agent-framework-mem0`](crates/agent-framework-mem0) | Mem0 hosted-API long-term-memory `ContextProvider`. |
| [`agent-framework-cosmos`](crates/agent-framework-cosmos) | Azure Cosmos DB NoSQL `ChatMessageStore` (master-key HMAC REST). |
| [`agent-framework-copilotstudio`](crates/agent-framework-copilotstudio) | Microsoft Copilot Studio agent client (Direct-to-Engine). |
| [`agent-framework-purview`](crates/agent-framework-purview) | Microsoft Purview compliance middleware (`processContent` DLP checks). |
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
| `azure` | [`agent-framework-azure`](crates/agent-framework-azure) — Azure OpenAI (api-key / Entra ID) | no |
| `mcp` | [`agent-framework-mcp`](crates/agent-framework-mcp) — Model Context Protocol tools (stdio, HTTP, websocket) | no |
| `a2a` | [`agent-framework-a2a`](crates/agent-framework-a2a) — Agent2Agent protocol client | no |
| `declarative` | [`agent-framework-declarative`](crates/agent-framework-declarative) — YAML/JSON agents & workflows | no |
| `hosting` | [`agent-framework-hosting`](crates/agent-framework-hosting) — serve agents over HTTP (DevUI-style, A2A, AG-UI, OpenAI-compatible) | no |
| `redis` | [`agent-framework-redis`](crates/agent-framework-redis) — Redis chat-message store & context provider | no |
| `mem0` | [`agent-framework-mem0`](crates/agent-framework-mem0) — Mem0 long-term memory provider | no |
| `azure-ai` | [`agent-framework-azure-ai`](crates/agent-framework-azure-ai) — Azure AI Foundry persistent agents | no |
| `azure-ai-search` | [`agent-framework-azure-ai-search`](crates/agent-framework-azure-ai-search) — Azure AI Search memory | no |
| `cosmos` | [`agent-framework-cosmos`](crates/agent-framework-cosmos) — Cosmos DB NoSQL message store | no |
| `copilotstudio` | [`agent-framework-copilotstudio`](crates/agent-framework-copilotstudio) — Copilot Studio agents | no |
| `purview` | [`agent-framework-purview`](crates/agent-framework-purview) — Purview compliance middleware | no |
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
| `azure_foundry_agent` | Azure AI Foundry persistent agents + `AzureCliCredential` | `AZURE_AI_PROJECT_ENDPOINT` + `az login`, `--features azure-ai,azure` |
| `copilotstudio_agent` | Copilot Studio agent over Direct-to-Engine | `COPILOTSTUDIOAGENT__*` vars + token, `--features copilotstudio` |
| `mcp_tools` | Connect an MCP stdio server, wire its tools into an agent | `OPENAI_API_KEY`, Node/`npx`, `--features mcp` |
| `mcp_sampling` | Answer MCP server-initiated sampling with your model | `OPENAI_API_KEY`, Node/`npx`, `--features mcp` |
| `hosting_server` | `AgentHost`: DevUI-style HTTP API + embedded debug page | `--features hosting` (offline with a canned agent) |
| `agui_server` | AG-UI protocol serving (`AgUiRouter` SSE events) | `--features hosting` (offline with a canned agent) |
| `a2a_client` | `A2AAgent` against any A2A server, incl. multi-turn continuity | `A2A_AGENT_URL`, `--features a2a` |
| `declarative_agent` | Load a `ChatAgent` from an inline YAML spec via `DeclarativeLoader` | `--features declarative` (offline with a canned client) |
| `redis_memory` | Redis-backed thread history + long-term memory provider | a Redis server, `--features redis` (skips gracefully) |
| `mem0_memory` | Mem0 long-term memory provider | `MEM0_API_KEY` + `OPENAI_API_KEY`, `--features mem0` |
| `cosmos_store` | Cosmos DB NoSQL conversation store (works on the emulator) | `COSMOS_ENDPOINT`/`COSMOS_KEY`, `--features cosmos` (skips gracefully) |
| `purview_middleware` | Purview `processContent` DLP checks around an agent | `PURVIEW_TOKEN` + `OPENAI_API_KEY`, `--features purview` |
| `retry_policy` | `RetryingChatClient` + `RetryPolicy` over a flaky client | offline — no key needed |
| `magentic_plan_review` | Magentic plan-review HITL: pause, revise, approve | offline — no key needed |
| `magentic_stall_intervention` | Magentic stall HITL: pause, replan with guidance | offline — no key needed |
| `workflow_checkpoint` | `FileCheckpointStorage`: save, list, resume mid-pipeline | offline — no key needed |
| `workflow_hitl` | `RequestInfoExecutor` pause / `send_response` resume | offline — no key needed |
| `workflow_viz` | Render a branching workflow as Mermaid and Graphviz DOT | offline — no key needed |

```bash
OPENAI_API_KEY=sk-... cargo run -p agent-framework --example quickstart
cargo run -p agent-framework --example retry_policy                  # offline
cargo run -p agent-framework --example magentic_stall_intervention   # offline
cargo run -p agent-framework --example hosting_server --features hosting
cargo run -p agent-framework --example agui_server --features hosting
ANTHROPIC_API_KEY=sk-ant-... cargo run -p agent-framework --example anthropic --features anthropic
```

## Design

The crate mirrors the Python framework's module structure so concepts map
one-to-one:

| Python (`agent_framework`) | Rust |
| --- | --- |
| `_types` | `types` (`ChatMessage`, `Content`, `ChatResponse`, `ChatOptions`, `ResponseFormat`, …) |
| `_clients` | `client` (`ChatClient`, `FunctionInvokingChatClient`, `RetryingChatClient`) |
| `_agents` | `agent` (`Agent`, `ChatAgent`, `as_tool`) |
| `_tools` | `tools` (`Tool`, `AiFunction`, hosted tools, `ApprovalMode`) |
| `_threads` | `threads` (`AgentThread`, `ChatMessageStore`) |
| `_memory` | `memory` (`ContextProvider`, `AggregateContextProvider`) |
| `_middleware` | `middleware` (agent / chat / function pipelines) |
| `observability` | `observability` (`ObservableChatClient`, OTel GenAI spans) |
| `_workflows` | `workflow` (`Workflow`, `WorkflowBuilder`, `Executor`, checkpointing, HITL, shared state, validation, viz, sub-workflows) |
| `_workflows._sequential` / `_concurrent` / `_group_chat` / `_handoff` / `_magentic` / `_agent` | `workflow::orchestration` (`SequentialBuilder`, `ConcurrentBuilder`, `GroupChatBuilder`, `HandoffBuilder`, `MagenticBuilder`, `WorkflowAgent`) |
| `_mcp` | [`agent-framework-mcp`](crates/agent-framework-mcp) (`McpStdioTool`, `McpStreamableHttpTool`, `McpWebsocketTool`, prompts, sampling, roots) |
| `a2a` package | [`agent-framework-a2a`](crates/agent-framework-a2a) (`A2AAgent`, `A2AClient`) |
| `declarative` package | [`agent-framework-declarative`](crates/agent-framework-declarative) (`DeclarativeLoader`) |
| `devui` / `ag-ui` packages | [`agent-framework-hosting`](crates/agent-framework-hosting) (`AgentHost`, `A2ARouter`, `AgUiRouter`, `OpenAiRouter`, embedded debug UI) |
| `redis` / `mem0` packages | [`agent-framework-redis`](crates/agent-framework-redis) / [`agent-framework-mem0`](crates/agent-framework-mem0) |
| `azure-ai` / `azure-ai-search` packages | [`agent-framework-azure-ai`](crates/agent-framework-azure-ai) / [`agent-framework-azure-ai-search`](crates/agent-framework-azure-ai-search) |
| `copilotstudio` / `purview` packages | [`agent-framework-copilotstudio`](crates/agent-framework-copilotstudio) / [`agent-framework-purview`](crates/agent-framework-purview) |
| (.NET `Microsoft.Agents.AI.CosmosNoSql`) | [`agent-framework-cosmos`](crates/agent-framework-cosmos) |

Cross-cutting behavior implemented in Python via class decorators
(`use_function_invocation`, `use_*_middleware`) is expressed in Rust as wrapper
types (`FunctionInvokingChatClient`, `RetryingChatClient`,
`ObservableChatClient`) and explicit middleware pipelines.

## Roadmap

See [PARITY.md](PARITY.md) for the full, grounded feature matrix. The
remaining gaps:

- [ ] MCP client: standalone GET-based SSE listening, automatic reconnect,
      elicitation
- [ ] A2A serving: push-notification config, `tasks/resubscribe`, and the
      authenticated extended card on the hosting side (the client has all
      three)
- [ ] The React DevUI frontend (an embedded single-file debug page ships
      today) and stateful hosted runs (conversation store, run-resume)
- [ ] Redis provider: embeddings/vector-KNN and hybrid search (BM25
      full-text ships on Redis Stack)
- [ ] Cosmos DB: Entra ID/AAD auth, `TransactionalBatch`, hierarchical
      partition keys, TTL
- [ ] The upstream Copilot-Studio declarative *workflow* DSL (declarative
      agents already follow the official schema)
- [ ] Purview: protection-scopes precheck/caching, background
      content-activity logging, JWT-derived identity fallback
- [ ] OTel SDK exporter wiring (spans are emitted and bridge-ready today)
- [ ] Remaining ecosystem: ChatKit, the `lab` experimental packages,
      DurableTask/Azure Functions hosting

Done — everything else, including:

- [x] Core data model, function-invocation loop, approval flow, structured
      output on every provider, and `RetryingChatClient` retries with
      `Retry-After` support
- [x] `ChatAgent` with threads, memory, `as_tool`, and agent / chat /
      function middleware
- [x] Providers: OpenAI (chat + Responses), Azure OpenAI, Azure AI Foundry,
      Anthropic, Copilot Studio; Entra ID credential chain
- [x] MCP client: stdio + HTTP + WebSocket, tools, prompts, sampling, roots
- [x] Workflow engine incl. signature-validated checkpointing, HITL, shared
      state, validation, viz, sub-workflows
- [x] All orchestrations incl. Magentic plan-review and stall-intervention
      HITL
- [x] A2A client (full task surface) and A2A serving (card + JSON-RPC)
- [x] Declarative YAML agents; Rust-native workflow specs
- [x] Hosting: DevUI-style API + embedded debug page, AG-UI, OpenAI-compatible
- [x] Redis (BM25), Mem0, Cosmos DB, Azure AI Search; Purview middleware
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
