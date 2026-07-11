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
| Azure OpenAI (Entra ID / bearer token) | ✅ | ✅ | ✅ done | `TokenCredential` trait + `StaticTokenCredential`, plus a hand-rolled Entra ID credential chain in `agent-framework-azure::credentials`: `AzureCliCredential` (shells `az account get-access-token`), `ClientSecretCredential` (OAuth2 client-credentials), `ManagedIdentityCredential` (IMDS), and `ChainedTokenCredential` (first success wins, remembered), all with per-scope token caching (2-min early refresh) and a `get_token_for_scope` scope override. Azure OpenAI's non-success path now maps to `Error::ServiceStatus` (+`Retry-After`) so `RetryingChatClient` works against it |
| Azure AI (Foundry) service client | ✅ (`azure-ai`) | ✅ (`Microsoft.Agents.AI.AzureAI[.Persistent]`) | ✅ done | `agent-framework-azure-ai::AzureAIAgentClient` — persistent-agents data plane spoken directly over REST (agents/threads/messages/runs, Assistants-style routes on the project endpoint with an `api-version` query param), SSE streaming (`thread.run.*`/`thread.message.delta`/`requires_action`) + non-streaming poll fallback, tool-call round-trip via `submit_tool_outputs` (call ids carry `[run_id, call_id]`), agent auto-create/delete lifecycle, `conversation_id`↔service thread id. Entra ID auth via `TokenCredential` (scope `https://ai.azure.com/.default`). Wire fidelity: routes/api-version follow the documented Assistants convention (the upstream wraps the `azure-ai-agents` SDK, so exact values aren't locally verifiable) — `api-version` is overridable |
| Anthropic Messages API | ✅ | ✅ | ✅ done | `agent-framework-anthropic::AnthropicClient`, hand-rolled (no Anthropic SDK dependency) |
| Anthropic structured output | ❌ (silently dropped) | ❌ (no `ResponseFormat` handling) | ✅ done | Rust *exceeds* parity here: `ResponseFormat::JsonObject`/`JsonSchema` is folded into the system prompt (the Messages API has no native `response_format`), see `convert::append_response_format_instructions` |
| `response.parse_json::<T>()` helper | ✅ (`response.value`) | ✅ | ✅ done | on both `ChatResponse` and `AgentRunResponse` |
| Retry / backoff policy layer | 🚧 (middleware pattern shown in docs, not built in) | not verified | ✅ done | `client::RetryingChatClient` wraps any `ChatClient` with a `RetryPolicy` (max_retries, exponential backoff, max_delay cap, jitter, `RetryOn` predicate). Default retries HTTP 408/429/5xx (`Error::ServiceStatus`) and transport-ish `Error::Service` failures; honors a server `Retry-After` (OpenAI + Anthropic now emit it on `ServiceStatus`) over computed backoff. Streaming retries only the initial connection |

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
| MCP prompts (`prompts/list` / `prompts/get`) | ✅ | not verified | ✅ done | `McpClient::list_prompts`/`::get_prompt`, and `.prompts()`/`.get_prompt(name, args)` on all three tool wrappers (mapping MCP `PromptMessage`s into core `ChatMessage`s, mirroring Python's `MCPTool.get_prompt`); `list_prompts`/`.prompts()` short-circuit to `[]` without a round trip when the server didn't declare the `prompts` capability |
| MCP sampling callback (server→client `sampling/createMessage`) | ✅ | not verified | ✅ done | `SamplingHandler` + `.sampling_handler(..)` on `McpClient` and all three tool wrappers; `chat_client_sampling_handler(client)` adapts any `ChatClient`; `sampling` capability declared in `initialize` only when a handler is set (matches the `mcp` Python SDK's capability-from-callback derivation); all three transports route a server-initiated request to the handler and write the JSON-RPC response back themselves — `ping` is always answered, an unhandled/unknown method gets a JSON-RPC "method not found" response, never silence |
| MCP roots callback (server→client `roots/list`) | 🚧 (the `mcp` Python SDK supports it; `agent_framework`'s `MCPTool` never wires up a callback, so it's unused in practice) | not verified | ✅ done | `.roots(vec![Root::new("file:///...")])` on `McpClient` and all three tool wrappers; static list only (no `list_changed` notifications, so `listChanged` is honestly advertised as `false`); the Rust port exceeds the upstream Python *package's* actual behavior here, though not the underlying protocol's |
| MCP remaining client surface (standalone GET-SSE listening, auto-reconnect, elicitation) | 🚧 (SDK-level) | not verified | ❌ not yet | the streamable-HTTP transport doesn't open the server-initiated GET `text/event-stream`; broken connections surface as errors instead of reconnecting; `elicitation/create` is answered with a JSON-RPC "method not found" (never silence) rather than implemented |

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
| Cosmos DB chat-message store | ❌ (no Python package) | ✅ (`Microsoft.Agents.AI.CosmosNoSql`) | 🚧 partial | `agent-framework-cosmos::CosmosChatMessageStore` — hand-rolled Cosmos DB NoSQL REST client (no `azure_data_cosmos`/`Microsoft.Azure.Cosmos` SDK dependency); one container, messages partitioned by `threadId`; `ensure_created()`, add/list/clear. **Master-key (HMAC-SHA256) auth only** — no Entra ID/AAD `TokenCredential` support, which the .NET package also offers; no `TransactionalBatch` (one `POST` per message on multi-message adds); no hierarchical partition keys; no TTL |

## Workflow engine

| Feature | Python | .NET | Rust | Notes |
| --- | --- | --- | --- | --- |
| Superstep (Pregel/BSP) graph engine | ✅ | ✅ | ✅ done | `workflow::{WorkflowBuilder, Workflow, WorkflowRun}`; `Single`/`FanOut`/`FanIn` edges, switch/case sugar |
| Checkpointing (in-memory + file-backed) | ✅ | ✅ | ✅ done | `CheckpointStorage` trait, `InMemoryCheckpointStorage`, `FileCheckpointStorage` (atomic write via temp+rename); fires automatically every superstep via `with_checkpointing` |
| Checkpoint graph-signature validation on resume | ✅ | ✅ | ✅ done | `WorkflowCheckpoint::graph_signature` records a deterministic FNV-1a fingerprint of the built graph (sorted executor ids + normalized edge-group descriptors + start; opaque conditions/selections contribute presence + declared labels only). `run_from_checkpoint` rejects a mismatch with an actionable `Error::Workflow` naming both signatures; `run_from_checkpoint_unchecked` overrides; legacy signatureless checkpoints resume with a `tracing` warn |
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
| Magentic human-in-the-loop stall intervention | ✅ (`MagenticHumanInterventionKind.STALL`, `with_human_input_on_stall`) | n/a (no Magentic orchestrator found) | ✅ done | `MagenticBuilder::with_stall_intervention()`: when `stall_count` exceeds `max_stall_count` the round loop pauses with a `MagenticStallInterventionRequest` (task, reason, stall/max counts, round, resets-so-far, facts/plan, last agent) instead of auto-replanning. `MagenticStallInterventionDecision` mirrors Python's stall decisions: `Continue` (=`CONTINUE`), `Replan { guidance }` (folds `REPLAN`+`GUIDANCE`: existing reset path, guidance appended to history), plus an added `Abort` (final-answer path). Disabled by default → unchanged auto-replan |
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
| A2A push notifications / task resubscribe / extended card | ✅ | ✅ | 🚧 partial | Client done: `A2AClient::set_push_notification_config`/`::get_push_notification_config` (`tasks/pushNotificationConfig/set`\|`/get` — note `get`'s params send the task id under `id`, not `taskId`, a real spec/SDK wire inconsistency faithfully preserved), `::resubscribe` (`tasks/resubscribe`, sharing its SSE parsing with `send_message_stream`), and `::get_extended_card` (`agent/getAuthenticatedExtendedCard`), auto-upgraded into `get_agent_card()` when `supportsAuthenticatedExtendedCard` is set (best-effort — falls back to the base card on failure). Serving side (`agent-framework-hosting::a2a::A2ARouter`) still exposes none of the three — out of scope for this change |
| DevUI-style HTTP API (entities + responses, SSE) | ✅ (`devui`) | ✅ (`Microsoft.Agents.AI.DevUI`) | ✅ done | `agent-framework-hosting::AgentHost` — `GET /health`, `GET /v1/entities[/{id}/info]`, `POST /v1/responses` (JSON or SSE); runs are stateless (no conversation store or run-resume endpoint) |
| DevUI web frontend | ✅ (bundled UI) | ✅ | 🚧 partial | not the React DevUI: `agent-framework-hosting::ui` embeds a single-file, dependency-free debug page (served at `GET /` and `GET /ui`) that lists entities and runs them against `POST /v1/responses`, rendering the SSE stream live (text deltas, collapsible executor/workflow rows, a resume-not-supported notice for pending `request_info`) |
| OpenAI-compatible serving (`/v1/chat/completions`) | ✅ (via devui) | ✅ (`.Hosting.OpenAI`) | ✅ done | `agent-framework-hosting::openai_compat::OpenAiRouter` (JSON or SSE) |
| Declarative agents (YAML/JSON specs) | ✅ (`declarative`) | ✅ (`Microsoft.Agents.AI.Declarative`) | ✅ done | `agent-framework-declarative::DeclarativeLoader::load_agent` — official schema vocabulary (`kind: Prompt`, `model.provider/apiType/options`, `tools`, `outputSchema`), `${VAR}` env interpolation, provider-agnostic `ChatClientFactory` / `ToolRegistry`; validated against real agent-samples specs in `tests/specs/` |
| Declarative workflows | ✅ (Power Platform / Copilot Studio DSL) | ✅ (`.Workflows.Declarative[.AzureAI]`) | 🚧 partial | `load_workflow` drives the graph engine + orchestration builders from a documented **Rust-native** `WorkflowSpec` (orchestration shorthand or node/edge graph); the upstream imperative Copilot-Studio DSL is intentionally not mapped |
| Hosting integrations (ASP.NET Core / Azure Functions / DurableTask) | n/a | ✅ (`Microsoft.Agents.AI.Hosting*`, `.DurableTask`) | ❌ not yet | the axum routers above are the Rust hosting story; no Azure Functions / DurableTask equivalents |
| AG-UI protocol | ✅ (`ag-ui`) | ✅ (`Microsoft.Agents.AI.AGUI`) | ✅ done | `agent-framework-hosting::agui::AgUiRouter` — `POST {path}` streaming camelCase SSE events with the SDK's exact SCREAMING_SNAKE `type` strings (`RUN_STARTED` → `TEXT_MESSAGE_START/CONTENT/END` and `TOOL_CALL_START/ARGS/END`/`TOOL_CALL_RESULT`/`CUSTOM` → `RUN_FINISHED`, or `RUN_ERROR`); `RunAgentInput` message/tool-call mapping mirrors `agui_messages_to_agent_framework`; frontend (client-declared) tool calls surface as `TOOL_CALL_*` without a result. Divergences: run-to-completion framing (the object-safe `Agent` trait has no streaming method), client `tools` accepted but not injected, and predictive-state (`STATE_*`/`MESSAGES_SNAPSHOT`) events omitted |
| CopilotStudio integration | ✅ (`copilotstudio`) | ✅ (`Microsoft.Agents.AI.CopilotStudio`) | ✅ done | `agent-framework-copilotstudio::CopilotStudioAgent` — a Direct-to-Engine (D2E) client speaking the wire protocol directly over `reqwest` (there is no Rust equivalent of the `microsoft-agents-copilotstudio-client` PyPI package to wrap). `CopilotStudioSettings` mirrors the `COPILOTSTUDIOAGENT__*` env vars; `CopilotStudioConnectionSettings` reproduces the reference SDK's environment-id/cloud host hashing and conversation-URL construction at **high fidelity** (built against that SDK's actual v1.1.0 source, not an inferred convention — see the crate docs for provenance) with a `direct_connect_url` host override; response parsing handles both SSE `event: activity`/`data:` frames and a bare JSON array. Real conversation-id continuity via `AgentThread` — Python's `CopilotStudioAgent.run` restarts the D2E conversation on *every* call, discarding prior context; this port persists and reuses it instead, mirroring `agent-framework-a2a`'s contextId/taskId fix. Auth is bring-your-own-token (`TokenProvider`); no MSAL/interactive-browser-login equivalent (see the crate docs' "Auth burden"). `run_stream` not implemented — the `Agent` trait has no streaming method, and Python's own `run_stream` only ever surfaces `typing` activity text anyway (never the final `message`), which this port's non-streaming `run` avoids by construction |
| Purview compliance middleware | ✅ (`purview`) | ✅ (`Microsoft.Agents.AI.Purview`) | ✅ done | `agent-framework-purview::{PurviewAgentMiddleware, PurviewChatMiddleware}` — evaluate the outgoing prompt and the agent/model response against Microsoft Graph `dataSecurityAndGovernance/processContent`, blocking either direction on a `blockAccess`/`block` DLP verdict and substituting a configurable system message; both directions evaluate as `Activity::UploadText`, faithfully matching a Python quirk (not `DownloadText` for the response direction) confirmed against Python's own test suite. Self-contained bring-your-own-token `TokenProvider` (no `azure-identity` dependency). 🚧 deliberate scope cut (per this work package's brief): calls only `processContent` — no `protectionScopes/compute` precheck/ETag caching or background `contentActivities` audit logging (Python's `ScopedContentProcessor`), and no bearer-token-JWT-derived `tenant_id`/app-location fallback (`PurviewSettings::tenant_id`/`purview_app_location` must be set explicitly) |
| Azure AI Search context provider | ✅ (`azure-ai-search`) | ✅ | ✅ done | `agent-framework-azure-ai-search::AzureAISearchProvider` implements core `ContextProvider`: `POST {endpoint}/indexes('{index}')/docs/search?api-version=2024-07-01` (api-key or `TokenCredential` bearer, scope `https://search.azure.com/.default`), query from the latest user message, configurable `top`/`select` fields/semantic configuration/vector query passthrough (server-side or client-side embedding); results fold into `Context` instructions with the `[Source: <id>]` citation + header convention. Ports the Python provider's *semantic* mode (its separate Knowledge-Base "agentic" mode is out of scope) |
| ChatKit / `lab` extras | ✅ (various packages) | ✅ (various packages) | ❌ not yet | not individually tracked in this matrix; CosmosDB has its own row under Middleware & memory (🚧 partial, master-key auth only) and Azure AI Search its own row above |
| Guardrails (dedicated module) | ❌ (middleware-based, no dedicated module) | ❌ (middleware-based) | ❌ not yet | none of the three implementations ship a first-class guardrails module; all three can express it via their middleware pipeline |

## Summary of remaining gaps

Everything not listed here ships (see the tables above). What genuinely remains:

- **MCP client**: no standalone GET-based SSE listening on the streamable-HTTP transport, no automatic reconnect, and no elicitation support (`elicitation/create` gets a JSON-RPC "method not found").
- **A2A serving**: the hosting router exposes no push-notification config, `tasks/resubscribe`, or authenticated extended card — those three ship on the *client* only.
- **DevUI frontend**: the embedded page is a single-file debug UI, not the React DevUI; hosted runs are stateless (no conversation store / run-resume endpoint).
- **Declarative workflows**: Rust-native spec, not the upstream Copilot-Studio imperative DSL (declarative *agents* follow the official schema).
- **Redis provider**: no embeddings/vector-KNN or hybrid search — retrieval is BM25 full-text on Redis Stack, SCAN+token-match+recency on plain Redis.
- **Cosmos DB store**: master-key (HMAC) auth only, no Entra ID/AAD; one `POST` per message instead of `TransactionalBatch`; no hierarchical partition keys or TTL.
- **Hosted tools** (code interpreter, web search, file search, hosted MCP): pass-through markers only.
- **Azure AI Foundry fidelity**: routes/`api-version` follow the documented Assistants convention rather than a locally verifiable SDK (the upstream wraps `azure-ai-agents`, which has no Rust equivalent).
- **Purview**: covers only `processContent` — no protection-scopes precheck/caching, background content-activity logging, or JWT-derived identity fallback.
- **Observability**: spans are OTel-GenAI-shaped and bridge-ready, but no OTel SDK exporter is wired up or shipped.
- **Ecosystem**: ChatKit, the `lab` experimental packages, DurableTask/Azure Functions hosting, and a dedicated guardrails module (which no implementation ships) remain unported.
