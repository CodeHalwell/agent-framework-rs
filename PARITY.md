# Feature parity: Rust vs. Python vs. .NET

A snapshot of `agent-framework-rs` against the upstream [Microsoft Agent
Framework](https://github.com/microsoft/agent-framework) (Python and .NET), as
of this port's current state. The **Rust** column is ground-truthed against
the source in this repository (grepped, not guessed); Python/.NET columns
reflect the reference implementation at the revision checked into
`/home/user/agent-framework` for this work.

Legend: ✅ done · 🚧 partial · ❌ not yet.

## Core types & content

| Feature | Python | .NET | Rust | Notes |
| --- | --- | --- | --- | --- |
| `ChatMessage` (role + content list) | ✅ | ✅ | ✅ done | `types::ChatMessage` |
| Unified `Content` union | ✅ | ✅ | ✅ done | `types::Content` — 12 variants: `Text`, `TextReasoning`, `Data`, `Uri`, `Error`, `FunctionCall`, `FunctionResult`, `Usage`, `HostedFile`, `HostedVectorStore`, `FunctionApprovalRequest`, `FunctionApprovalResponse` (`crates/agent-framework-core/src/types/content.rs`) |
| Function-call / function-result content | ✅ | ✅ | ✅ done | includes streamed-fragment merge logic |
| Function-approval request/response content | ✅ | ✅ | ✅ done | the HITL approval primitive |
| `ChatResponse` + streaming aggregation | ✅ | ✅ | ✅ done | `ChatResponse::absorb_update` / `::finalize` (`types/response.rs`) |
| `ChatOptions` + merge semantics | ✅ | ✅ | ✅ done | `ChatOptions::merge` mirrors Python's `&` operator |
| `AgentRunResponse` / `AgentRunResponseUpdate` | ✅ | ✅ | ✅ done | |
| `FinishReason` (open string enum) | ✅ | ✅ | ✅ done | |
| `ToolMode` (auto/required/none) | ✅ | ✅ | ✅ done | |
| `UsageDetails` (+ accumulation) | ✅ | ✅ | ✅ done | |
| `ResponseFormat` (structured-output request) | ✅ | ✅ | ✅ done | `Text` / `JsonObject` / `JsonSchema{..}`, `ResponseFormat::json_schema(name, schema)` |

## Chat clients & providers

| Feature | Python | .NET | Rust | Notes |
| --- | --- | --- | --- | --- |
| `ChatClient` trait | ✅ | ✅ | ✅ done | `client::ChatClient` |
| Automatic function-invocation loop | ✅ | ✅ | ✅ done | `client::FunctionInvokingChatClient` — concurrent tool execution, approval gating, max-iterations failsafe |
| OpenAI Chat Completions | ✅ | ✅ | ✅ done | `agent-framework-openai::OpenAIClient` |
| OpenAI Responses API | ✅ | ✅ | ✅ done | `OpenAIResponsesClient`; `conversation_id` ↔ `previous_response_id` |
| Azure OpenAI (API key) | ✅ | ✅ | ✅ done | `agent-framework-azure::AzureOpenAIClient::new` |
| Azure OpenAI (Entra ID / bearer token) | ✅ | ✅ | 🚧 partial | `TokenCredential` trait + `StaticTokenCredential` (fixed token) ship; a real Entra ID credential chain (wrapping e.g. `azure_identity`) is left to the caller |
| Azure AI (Foundry) service client | ✅ (`azure-ai`) | ✅ (`Microsoft.Agents.AI.AzureAI[.Persistent]`) | ❌ not yet | only Azure *OpenAI* is covered |
| Anthropic Messages API | ✅ | ✅ | ✅ done | `agent-framework-anthropic::AnthropicClient`, hand-rolled (no Anthropic SDK dependency) |
| Anthropic structured output | ❌ (silently dropped) | ❌ (no `ResponseFormat` handling) | ✅ done | Rust *exceeds* parity here: `ResponseFormat::JsonObject`/`JsonSchema` is folded into the system prompt (the Messages API has no native `response_format`), see `convert::append_response_format_instructions` |
| `response.parse_json::<T>()` helper | ✅ (`response.value`) | ✅ | ✅ done | on both `ChatResponse` and `AgentRunResponse` |
| Retry / backoff policy layer | 🚧 (middleware pattern shown in docs, not built in) | not verified | ❌ not yet | no provider retries a failed request; a caller-supplied middleware or wrapper `ChatClient` would need to add one |

## Agents

| Feature | Python | .NET | Rust | Notes |
| --- | --- | --- | --- | --- |
| `Agent` trait | ✅ | ✅ | ✅ done | `agent::Agent` |
| `ChatAgent` (+ builder) | ✅ | ✅ | ✅ done | instructions, default options, tools, context providers, agent/chat/function middleware |
| `AgentThread` (service-managed XOR local) | ✅ | ✅ | ✅ done | `threads::AgentThread` |
| `ChatMessageStore` / `InMemoryChatMessageStore` | ✅ | ✅ | ✅ done | |
| `agent.as_tool()` | ✅ | ✅ | ✅ done | `ChatAgent::as_tool` + `AsToolOptions`; runs the wrapped agent statelessly |
| Workflow-as-agent | ✅ | ✅ | ✅ done | `orchestration::workflow_agent::WorkflowAgent`, `WorkflowAgentExt::as_agent` |
| WorkflowAgent thread-history updates | ✅ | ✅ | ✅ done | `run` / `run_stream_with_thread` write input + response back to the thread (mirrors Python's `_notify_thread_of_new_messages`); like Python, write-back only — prior history is not fed into the workflow input |
| A2A remote agent (`A2AAgent`) | ✅ (`a2a` package) | ✅ (`Microsoft.Agents.AI.A2A`) | ✅ done | `agent-framework-a2a::A2AAgent` — JSON-RPC 2.0 over HTTP + `.well-known` card discovery; thread-based `contextId`/`taskId` continuity (which the Python client actually lacks) |

## Tools & MCP

| Feature | Python | .NET | Rust | Notes |
| --- | --- | --- | --- | --- |
| `Tool` trait / `AiFunction` | ✅ | ✅ | ✅ done | `tools::Tool`, `tools::AiFunction` (closure-backed) |
| Hosted tool markers (code interpreter, web search, file search, MCP) | ✅ (executes server-side) | ✅ | 🚧 partial | `hosted_*` constructors are pass-through markers only (`executor: None`) — the provider must support them server-side; there is no local emulation |
| `ApprovalMode` / approval flow | ✅ | ✅ | ✅ done | enforced in `FunctionInvokingChatClient::get_response` |
| MCP client — stdio transport | ✅ | ✅ (via `ModelContextProtocol` SDK integration) | ✅ done | `agent-framework-mcp::McpStdioTool` |
| MCP client — streamable HTTP transport | ✅ | ✅ | ✅ done | `McpStreamableHttpTool` |
| MCP client — WebSocket transport | ✅ | not verified | ✅ done | `McpWebsocketTool` (`ws://`/`wss://`, `"mcp"` subprotocol, one JSON-RPC message per text frame) |
| MCP prompts (`prompts/list` / `prompts/get`) | ✅ | not verified | ❌ not yet | only tools are exposed as agent functions |
| MCP sampling / roots callbacks | ✅ | not verified | ❌ not yet | server-initiated requests are logged and ignored, never answered |

## Middleware & memory

| Feature | Python | .NET | Rust | Notes |
| --- | --- | --- | --- | --- |
| Agent middleware pipeline | ✅ | ✅ | ✅ done | `middleware::{Middleware, MiddlewarePipeline, Next}`, wired into `ChatAgent::run`/`run_stream` |
| Chat middleware pipeline | ✅ | ✅ | ✅ done | `ChatAgentBuilder::chat_middleware` — runs around every underlying chat-client call (mirrors `use_chat_middleware`); pre-call-only on the token-streaming path, per the module docs |
| Function-invocation middleware pipeline | ✅ | ✅ | ✅ done | `ChatAgentBuilder::function_middleware`, plumbed into `FunctionInvokingChatClient::with_function_middleware` and run around every local tool call |
| `ContextProvider` / `AggregateContextProvider` | ✅ | ✅ | ✅ done | `memory::ContextProvider`, fan-out/merge over multiple providers |
| Mem0-backed memory provider | ✅ (`mem0` package, wraps the `mem0` SDK) | ✅ (`Microsoft.Agents.AI.Mem0`) | ✅ done | `agent-framework-mem0::Mem0Provider` — direct REST (`/v1/memories/`, `/v2/memories/search/`), scoped by application/agent/user/thread id |
| Redis chat-message store | ✅ (`redis` package) | not found | ✅ done | `agent-framework-redis::RedisChatMessageStore` — one LIST per thread, JSON messages, optional trimming; close mirror of Python |
| Redis context provider (long-term memory) | ✅ (RediSearch: full-text + vector/hybrid) | not found | 🚧 partial | `RedisContextProvider` ports the *scoping* semantics; when the connected server has RediSearch loaded (Redis Stack) it now manages a real `FT.CREATE ... ON JSON` index and serves `invoking()` via `FT.SEARCH` (BM25-ranked, TAG-filtered, `LIMIT`ed), falling back to the original SCAN+token-match+recency path on plain Redis or when `with_force_scan_fallback` is set; still no embeddings/vector or hybrid search (documented divergence in `context_provider.rs`) |
| Cosmos DB chat-message store | ✅ (`azure-ai-projects`/CosmosDB-adjacent, not individually tracked) | ✅ (`Microsoft.Agents.AI.CosmosNoSql`) | 🚧 partial | `agent-framework-cosmos::CosmosChatMessageStore` — hand-rolled Cosmos DB NoSQL REST client (no `azure_data_cosmos`/`Microsoft.Azure.Cosmos` SDK dependency); one container, messages partitioned by `threadId`; `ensure_created()`, add/list/clear. **Master-key (HMAC-SHA256) auth only** — no Entra ID/AAD `TokenCredential` support, which the .NET package also offers; no `TransactionalBatch` (one `POST` per message on multi-message adds); no hierarchical partition keys; no TTL |

## Workflow engine

| Feature | Python | .NET | Rust | Notes |
| --- | --- | --- | --- | --- |
| Superstep (Pregel/BSP) graph engine | ✅ | ✅ | ✅ done | `workflow::{WorkflowBuilder, Workflow, WorkflowRun}`; `Single`/`FanOut`/`FanIn` edges, switch/case sugar |
| Checkpointing (in-memory + file-backed) | ✅ | ✅ | ✅ done | `CheckpointStorage` trait, `InMemoryCheckpointStorage`, `FileCheckpointStorage` (atomic write via temp+rename); fires automatically every superstep via `with_checkpointing` |
| Checkpoint graph-signature validation on resume | ✅ | ✅ | ❌ not yet | `run_from_checkpoint` trusts the caller: it never checks that the resuming graph matches the checkpointed one, and silently skips state for unknown executor ids |
| Request/response HITL (`request_info`) | ✅ | ✅ | ✅ done | `WorkflowContext::request_info`, `RequestInfoExecutor`, `WorkflowRun::send_response(s)` |
| Shared state | ✅ | ✅ | ✅ done | `workflow::SharedState`, cloned into every `WorkflowContext`, checkpointed automatically |
| Graph validation (structural) | ✅ | ✅ | ✅ done | start registered, unknown-executor edges, duplicate edges, start-reachability — run automatically in `WorkflowBuilder::build`. Python's additional *static type-compatibility* checks are intentionally out of scope for a `serde_json::Value`-typed engine |
| Visualization (Mermaid + Graphviz DOT) | ✅ | ✅ | ✅ done | `Workflow::viz().to_mermaid()` / `.to_dot()` |
| Sub-workflow composition | ✅ | ✅ | ✅ done | `workflow::WorkflowExecutor` wraps a child `Workflow` as a parent-graph node; documented divergence from Python's `SubWorkflow{Request,Response}Message` wrapper shape |

## Orchestrations

| Feature | Python | .NET | Rust | Notes |
| --- | --- | --- | --- | --- |
| Sequential | ✅ | ✅ | ✅ done | `orchestration::SequentialBuilder` |
| Concurrent (fan-out/fan-in) | ✅ | ✅ | ✅ done | `orchestration::ConcurrentBuilder` |
| Group chat — custom / LLM manager | ✅ | ✅ | ✅ done | `GroupChatBuilder::manager` / `::manager_fn` / `::manager_agent` |
| Group chat — round-robin manager | ❌ (no built-in; requires `set_manager`/`set_select_speakers_func`) | ✅ (`RoundRobinGroupChatManager`) | ✅ done | `RoundRobinManager`, the `GroupChatBuilder` default |
| Handoff | ✅ | ✅ | ✅ done | `orchestration::HandoffBuilder`; autonomous and human-in-loop interaction modes |
| Magentic (plan / progress-ledger / final answer) | ✅ | not found | ✅ done | `MagenticBuilder` + `StandardMagenticManager`; prompts ported verbatim from Python |
| Magentic human-in-the-loop plan review | ✅ (`MagenticHumanInterventionKind.PLAN_REVIEW`) | n/a (no Magentic orchestrator found) | ✅ done | `MagenticBuilder::with_plan_review()` + `max_plan_review_rounds(n)`: pauses after `plan()` with a `MagenticPlanReviewRequest` (task/facts/plan/round), answered by a `MagenticPlanReviewDecision` — approve, revise-with-comments (triggers `replan` with the feedback in `chat_history`), or revise-with-edited-plan (adopted verbatim, no LLM call) |
| Workflow-as-agent | ✅ | ✅ | ✅ done | see Agents section |

## Observability

| Feature | Python | .NET | Rust | Notes |
| --- | --- | --- | --- | --- |
| GenAI-semantic-convention tracing spans | ✅ | ✅ | ✅ done | `observability::ObservableChatClient` (`chat` span); `invoke_agent` span in `ChatAgent::run_core`; `execute_tool` span in the function-invocation loop |
| OpenTelemetry SDK exporter wiring | ✅ | ✅ | 🚧 partial | spans follow OTel GenAI conventions and are bridge-ready (e.g. via `tracing-opentelemetry`), but no OTel SDK/exporter is wired up or shipped |

## Serving & ecosystem

| Feature | Python | .NET | Rust | Notes |
| --- | --- | --- | --- | --- |
| A2A client (call remote agents) | ✅ (`a2a`) | ✅ (`Microsoft.Agents.AI.A2A`) | ✅ done | `agent-framework-a2a` — see Agents section |
| A2A serving (agent card + JSON-RPC endpoint) | ✅ | ✅ (`.Hosting.A2A[.AspNetCore]`) | ✅ done | `agent-framework-hosting::a2a::A2ARouter` — `GET /.well-known/agent-card.json` + JSON-RPC 2.0 `POST /` |
| A2A push notifications / task resubscribe / extended card | ✅ | ✅ | ❌ not yet | `tasks/pushNotificationConfig/*`, `tasks/resubscribe`, and the authenticated extended-card flow are unimplemented on both the client and serving sides |
| DevUI-style HTTP API (entities + responses, SSE) | ✅ (`devui`) | ✅ (`Microsoft.Agents.AI.DevUI`) | ✅ done | `agent-framework-hosting::AgentHost` — `GET /health`, `GET /v1/entities[/{id}/info]`, `POST /v1/responses` (JSON or SSE); runs are stateless (no conversation store or run-resume endpoint) |
| DevUI web frontend | ✅ (bundled UI) | ✅ | ❌ not yet | this port ships the API surface only, no bundled browser UI |
| OpenAI-compatible serving (`/v1/chat/completions`) | ✅ (via devui) | ✅ (`.Hosting.OpenAI`) | ✅ done | `agent-framework-hosting::openai_compat::OpenAiRouter` (JSON or SSE) |
| Declarative agents (YAML/JSON specs) | ✅ (`declarative`) | ✅ (`Microsoft.Agents.AI.Declarative`) | ✅ done | `agent-framework-declarative::DeclarativeLoader::load_agent` — official schema vocabulary (`kind: Prompt`, `model.provider/apiType/options`, `tools`, `outputSchema`), `${VAR}` env interpolation, provider-agnostic `ChatClientFactory` / `ToolRegistry`; validated against real agent-samples specs in `tests/specs/` |
| Declarative workflows | ✅ (Power Platform / Copilot Studio DSL) | ✅ (`.Workflows.Declarative[.AzureAI]`) | 🚧 partial | `load_workflow` drives the graph engine + orchestration builders from a documented **Rust-native** `WorkflowSpec` (orchestration shorthand or node/edge graph); the upstream imperative Copilot-Studio DSL is intentionally not mapped |
| Hosting integrations (ASP.NET Core / Azure Functions / DurableTask) | n/a | ✅ (`Microsoft.Agents.AI.Hosting*`, `.DurableTask`) | ❌ not yet | the axum routers above are the Rust hosting story; no Azure Functions / DurableTask equivalents |
| AG-UI protocol | ✅ (`ag-ui`) | ✅ (`Microsoft.Agents.AI.AGUI`) | ❌ not yet | |
| CopilotStudio integration | ✅ (`copilotstudio`) | ✅ (`Microsoft.Agents.AI.CopilotStudio`) | ❌ not yet | |
| Purview / ChatKit / Azure AI Search / `lab` extras | ✅ (various packages) | ✅ (various packages) | ❌ not yet | not individually tracked in this matrix; CosmosDB has moved to its own row under Middleware & memory (🚧 partial, master-key auth only) |
| Guardrails (dedicated module) | ❌ (middleware-based, no dedicated module) | ❌ (middleware-based) | ❌ not yet | none of the three implementations ship a first-class guardrails module; all three can express it via their middleware pipeline |

## Summary of remaining gaps

Much shorter than it used to be. What genuinely remains:

- **MCP**: prompts capability and sampling/roots callbacks (the WebSocket transport now ships).
- **A2A**: push notifications, `tasks/resubscribe`, authenticated extended card.
- **DevUI**: no bundled web frontend; hosted runs are stateless (no conversation store / run-resume endpoint).
- **Declarative workflows**: Rust-native spec, not the upstream Copilot-Studio imperative DSL (declarative *agents* follow the official schema).
- **Redis provider**: `FT.SEARCH` BM25 full-text now backs retrieval on Redis Stack (with a SCAN+token-match+recency fallback on plain Redis); vector/hybrid search is still not ported.
- **Cosmos DB store**: master-key (HMAC) auth only, no Entra ID/AAD; one `POST` per message instead of `TransactionalBatch`; no hierarchical partition keys or TTL.
- **Hosted tools** (code interpreter, web search, file search, hosted MCP): pass-through markers only.
- **Reliability**: no built-in retry/backoff policy layer for any provider.
- **Checkpointing**: no graph-signature validation on resume.
- **Azure**: no Azure AI (Foundry) service client, and no real Entra ID credential chain (bring your own `TokenCredential`) — the same gap shows up in the new Cosmos DB store.
- **Ecosystem**: AG-UI, CopilotStudio, Purview, ChatKit, Azure AI Search, DurableTask/Azure Functions hosting, OTel SDK exporter wiring.
