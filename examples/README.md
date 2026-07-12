# agent-framework-rs examples

Runnable examples for `agent-framework-rs`, organized by topic under one
`agent-framework-examples` crate (mirrors the layout of the upstream
[Microsoft Agent Framework](https://github.com/microsoft/agent-framework)'s
own samples tree).

## Running an example

Every example is a `[[example]]` target in [`Cargo.toml`](Cargo.toml), so all
of them run the same way from the repo root:

```bash
cargo run -p agent-framework-examples --example <name>
```

Some need credentials for a real provider or backing service, passed as
environment variables:

```bash
OPENAI_API_KEY=sk-... cargo run -p agent-framework-examples --example quickstart
```

No example needs a `--features` flag: the crate depends on `agent-framework`
with `features = ["full", "otel-metrics"]`, so every provider/integration is
already compiled in.

**Offline vs. needs keys.** Each table below has a `Requires` column.
`offline` means the example runs with no credentials, network access, or
external service at all. Anything else names the environment variable(s) (or
external service) it needs. Most credentialed examples **skip gracefully**
(print a one-line message and exit `0`) when their env var(s) are unset,
rather than failing — the per-example note says so; where it doesn't, the
example fails outright without the credential (the same behavior as
`quickstart`). A few examples (`redis_memory`, `cosmos_store`) instead check
that a real backing service is reachable and skip gracefully if it isn't.

**MCP examples** (`mcp_*`) additionally need a working `npx` (Node.js) on
`PATH` — they spawn `@modelcontextprotocol/server-everything`, the MCP
project's own reference/test server, on first use (which needs network access
the very first time, to let npm fetch and cache the package).

**Hosting examples** (`*_server`, `streaming_sse`) serve HTTP: they print the
address and a sample `curl` command, then serve until you hit Ctrl-C (except
`streaming_sse`, which drives one request against its own in-process server
and exits on its own — no second terminal needed).

## Agents (`agents/`)

Core `Agent` mechanics: building, running, streaming, tools, threads, and
middleware.

| Example | Shows | Requires |
| --- | --- | --- |
| `quickstart` | Minimal `Agent` + OpenAI in a few lines | `OPENAI_API_KEY` |
| `streaming` | Token-by-token agent streaming | `OPENAI_API_KEY` |
| `tools` | Local tool calling via the automatic function-invocation loop | `OPENAI_API_KEY` |
| `typed_tools` | `FunctionTool::typed` derives a JSON Schema from a `#[derive(JsonSchema)]` struct | offline (live model optional, `OPENAI_API_KEY`) |
| `structured_output` | `ResponseFormat::json_schema` + `response.parse_json::<T>()` | `OPENAI_API_KEY` |
| `approvals` | Human-in-the-loop tool approval: pause, inspect, approve, resume | `OPENAI_API_KEY` |
| `agent_as_tool` | Compose agents: a specialist exposed as a tool via `Agent::as_tool` | `OPENAI_API_KEY` |
| `retry_policy` | `RetryingChatClient` + `RetryPolicy` over a scripted flaky client | offline |
| `per_run_options` | `AgentRunOptions` merges per-run `ChatOptions` overrides over the agent's defaults | offline |
| `thread_persistence` | `thread.serialize()` / `Agent::deserialize_thread` round-trip a conversation | offline |
| `multi_turn_conversation` | One `AgentThread` reused across calls accumulates history automatically | offline |
| `image_input` | Attach an image via `Content::Uri` (URL) or `Content::Data` (inline bytes) | `OPENAI_API_KEY` (vision model; skips gracefully) |
| `agent_middleware` | Wrap a whole agent run: logging plus early-termination middleware | offline |
| `function_middleware` | Wrap every local tool call: rewrite arguments, observe/override results | offline |
| `chat_middleware` | Wrap the underlying `ChatClient` call itself, one level below agent middleware | offline |
| `custom_context_provider` | A `ContextProvider` with `invoking` / `thread_created` / `invoked` hooks | offline |

## Providers (`providers/`)

Every `ChatClient` backend: OpenAI (Chat + Responses), Azure
OpenAI, Azure AI Foundry, Anthropic, and Copilot Studio.

| Example | Shows | Requires |
| --- | --- | --- |
| `openai_responses` | OpenAI Responses API + `conversation_id` (`previous_response_id`) reuse | `OPENAI_API_KEY` |
| `openai_compatible_endpoint` | `OpenAIClient` against any OpenAI-Chat-compatible server (llama.cpp, Ollama, vLLM, ...) | `OPENAI_BASE_URL` |
| `anthropic` | The Anthropic (Claude) Messages API client | `ANTHROPIC_API_KEY` |
| `anthropic_hosted_tools` | Anthropic hosted web-search / code-execution tools (server-side, no local wiring) | `ANTHROPIC_API_KEY` (skips gracefully) |
| `azure_openai` | Azure OpenAI with both api-key and Entra ID (`TokenCredential`) auth | `AZURE_OPENAI_*` |
| `azure_openai_responses` | `AzureOpenAIResponsesClient`: the Responses API on Azure OpenAI | `AZURE_OPENAI_*` (skips gracefully) |
| `azure_default_credential` | `DefaultAzureCredential`'s four-link Entra ID credential chain | `AZURE_OPENAI_ENDPOINT` (+ `az login`; skips gracefully) |
| `azure_foundry_agent` | Azure AI Foundry persistent agents (Assistants-style REST) via `AzureAIAgentClient` | `AZURE_AI_PROJECT_ENDPOINT` (+ `az login`; skips gracefully) |
| `azure_foundry_bing_grounding` | Bing grounding on Azure AI Foundry via a connection id | `AZURE_AI_PROJECT_ENDPOINT`, `BING_CONNECTION_ID` (skips gracefully) |
| `copilotstudio_agent` | Microsoft Copilot Studio agent over the Direct-to-Engine protocol | `COPILOTSTUDIOAGENT__*` + token (skips gracefully) |

## Workflows (`workflows/`)

The Pregel/BSP graph engine: executors, edges, fan-out/fan-in, switch/case,
loops, checkpointing, shared state, sub-workflows, and custom executors.

| Example | Shows | Requires |
| --- | --- | --- |
| `fan_out_fan_in` | One source dispatches to three executors; an aggregator waits for all three | offline |
| `conditional_edges` | `add_conditional_edge` fires every edge whose condition holds (no exclusivity) | offline |
| `switch_case` | `add_switch` routes to the first matching case, else the default | offline |
| `loops_and_max_iterations` | A self-looping executor, plus `set_max_iterations` as the runaway-loop guard | offline |
| `shared_state` | Pass a lightweight reference through the graph; payload + audit trail live in `SharedState` | offline |
| `sub_workflows` | `WorkflowExecutor` embeds a whole child `Workflow` as one parent-graph node | offline |
| `agents_in_workflows` | Mix `AgentExecutor` (wraps an `Agent`) with `FunctionExecutor` nodes in one graph | offline |
| `workflow_as_agent` | Expose a built `Workflow` as an `Agent` via `WorkflowAgentExt::as_agent` | offline |
| `custom_executors` | A hand-written `Executor` (custom snapshot/restore) vs. a `FunctionExecutor` closure | offline |
| `concurrent_supersteps` | Same-superstep executors run concurrently via `join_all`, not sequentially | offline |
| `checkpoint_resume_fanin` | Checkpoint mid-way through a partially-satisfied fan-in barrier, then resume | offline |
| `workflow_checkpoint` | `FileCheckpointStorage`: persist every superstep, list, and resume mid-pipeline | offline |
| `workflow_hitl` | `RequestInfoExecutor` pause / `send_response` resume | offline |
| `workflow_viz` | Render a branching workflow as Mermaid and Graphviz DOT | offline |

## Orchestrations (`orchestrations/`)

Prebuilt multi-agent patterns on top of the workflow engine: sequential,
concurrent, group chat, handoff, and Magentic (with HITL plan-review and
stall-intervention).

| Example | Shows | Requires |
| --- | --- | --- |
| `sequential` | A writer drafts, then an editor revises (`SequentialBuilder`) | `OPENAI_API_KEY` |
| `concurrent` | `ConcurrentBuilder` fans one prompt out to several agents and aggregates replies | offline |
| `group_chat` | Multi-agent group chat: round-robin turn-taking or an LLM "manager" | `OPENAI_API_KEY` |
| `handoff` | A triage agent hands off to specialists via a synthetic tool call | `OPENAI_API_KEY` |
| `magentic` | `StandardMagenticManager` plans, assigns work each round, drafts a final answer | `OPENAI_API_KEY` |
| `magentic_plan_review` | Magentic HITL: pause after the initial plan, approve/revise it | offline |
| `magentic_stall_intervention` | Magentic HITL: pause on a detected stall, continue/replan/abort | offline |
| `streaming_updates` | `run_stream` over a workflow surfaces each agent's reply incrementally | offline |

## MCP (`mcp/`)

Model Context Protocol: tools (static and dynamic), prompts, server-initiated
sampling, and roots. All five connect to
`@modelcontextprotocol/server-everything` over stdio.

| Example | Shows | Requires |
| --- | --- | --- |
| `mcp_tools` | Connect an MCP stdio server, list its tools, wire them into a `Agent` | `OPENAI_API_KEY`, `npx` |
| `mcp_first_class_tools` | `AgentBuilder::tool_source`: resolve an MCP server's tools fresh on every run | `OPENAI_API_KEY` (skips gracefully), `npx` |
| `mcp_prompts` | List an MCP server's prompts, render one, and run it through a real agent | `OPENAI_API_KEY` (skips gracefully), `npx` |
| `mcp_roots` | Advertise filesystem roots via `.roots(...)`; explains the server-side `roots/list` flow | `OPENAI_API_KEY` (skips gracefully), `npx` |
| `mcp_sampling` | Answer MCP server-initiated `sampling/createMessage` with your own model | `OPENAI_API_KEY` (skips gracefully), `npx` |

## Hosting (`hosting/`)

Serve agents over HTTP with `axum`: DevUI-style, A2A, OpenAI-compatible, and
AG-UI surfaces, all nestable into one app.

| Example | Shows | Requires |
| --- | --- | --- |
| `hosting_server` | `AgentHost`: DevUI-style HTTP API + embedded debug page | offline (canned fallback; `OPENAI_API_KEY` optional) |
| `agui_server` | AG-UI protocol serving (`AgUiRouter` camelCase SSE events) | offline (canned fallback; `OPENAI_API_KEY` optional) |
| `openai_compat_server` | `OpenAiRouter`: OpenAI-Chat-Completions-compatible `/v1/chat/completions` | offline (canned fallback; `OPENAI_API_KEY` optional) |
| `a2a_server` | `A2ARouter`: agent card + JSON-RPC `message/send` / `tasks/get` / `tasks/cancel` | offline (canned fallback; `OPENAI_API_KEY` optional) |
| `streaming_sse` | Real token streaming end to end: canned streaming agent + in-process SSE client | offline (self-terminating) |

## Memory (`memory/`)

Long-term memory and conversation-store backends: Redis, Mem0, Cosmos DB, and
Azure AI Search.

| Example | Shows | Requires |
| --- | --- | --- |
| `redis_memory` | Redis-backed thread history (`RedisChatMessageStore`) + long-term memory provider | a local Redis server (skips gracefully) |
| `mem0_memory` | Hosted Mem0 long-term memory: persist and retrieve memories per user | `MEM0_API_KEY`, `OPENAI_API_KEY` (skips gracefully) |
| `cosmos_store` | Azure Cosmos DB (NoSQL) conversation store (works against the emulator too) | `COSMOS_ENDPOINT`, `COSMOS_KEY` (skips gracefully) |
| `azure_ai_search` | Azure AI Search hybrid/semantic search as a long-term-memory `ContextProvider` | `AZURE_SEARCH_*`, `OPENAI_API_KEY` (skips gracefully) |

## Observability (`observability/`)

`tracing`-based OpenTelemetry GenAI semantic conventions: spans and metrics.

| Example | Shows | Requires |
| --- | --- | --- |
| `observability` | `ObservableChatClient` emits OTel GenAI-semantic-convention `tracing` spans | `OPENAI_API_KEY` |
| `otel_metrics` | GenAI token-usage/duration histograms via the `otel-metrics` feature | offline |

## A2A (`a2a/`)

The Agent2Agent protocol *client* side (see Hosting above for serving one).

| Example | Shows | Requires |
| --- | --- | --- |
| `a2a_client` | `A2AAgent`: talk to a remote A2A server as a local `Agent`, multi-turn continuity | `A2A_AGENT_URL` (e.g. the `a2a_server` example) |

## Declarative (`declarative/`)

Load agents and workflows from declarative specs instead of Rust builder
calls.

| Example | Shows | Requires |
| --- | --- | --- |
| `declarative_agent` | Load a `Agent` from an official-schema YAML spec via `DeclarativeLoader` | offline (canned fallback; `OPENAI_API_KEY` optional) |
| `declarative_workflow` | Load a `Workflow` from a Rust-native spec: orchestration shorthand and an explicit graph | offline |

## Compliance (`compliance/`)

| Example | Shows | Requires |
| --- | --- | --- |
| `purview_middleware` | `PurviewAgentMiddleware`: Graph `processContent` DLP checks around an agent | `PURVIEW_TOKEN`, `OPENAI_API_KEY` (skips gracefully) |

---

69 examples total. See the root [`README.md`](../README.md) for the
project-level overview, feature-flag table, and workspace layout.
