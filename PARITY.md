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
| Anthropic Messages API | ✅ | ✅ | ✅ done | `agent-framework-anthropic::AnthropicClient`, hand-rolled (no Anthropic SDK dependency) |
| Anthropic structured output | ✅ | ✅ | ❌ not yet | `ChatOptions::response_format` is silently ignored by `AnthropicClient` — `convert::build_request` never reads it |
| `response.parse_json::<T>()` helper | ✅ (`response.value`) | ✅ | ✅ done | on both `ChatResponse` and `AgentRunResponse` |
| Retry / backoff policy layer | 🚧 (middleware pattern shown in docs, not built in) | not verified | ❌ not yet | no provider (`OpenAIClient`, `AnthropicClient`, `AzureOpenAIClient`) retries a failed request; a caller-supplied middleware or wrapper `ChatClient` would need to add one |

## Agents

| Feature | Python | .NET | Rust | Notes |
| --- | --- | --- | --- | --- |
| `Agent` trait | ✅ | ✅ | ✅ done | `agent::Agent` |
| `ChatAgent` (+ builder) | ✅ | ✅ | ✅ done | instructions, default options, tools, context providers, agent middleware |
| `AgentThread` (service-managed XOR local) | ✅ | ✅ | ✅ done | `threads::AgentThread` |
| `ChatMessageStore` / `InMemoryChatMessageStore` | ✅ | ✅ | ✅ done | |
| `agent.as_tool()` | ✅ | ✅ | ✅ done | `ChatAgent::as_tool` + `AsToolOptions`; runs the wrapped agent statelessly |
| Workflow-as-agent | ✅ | ✅ | ✅ done | `orchestration::workflow_agent::WorkflowAgent`, `WorkflowAgentExt::as_agent` |
| WorkflowAgent thread-history updates | ✅ | ✅ | ❌ not yet | `WorkflowAgent::run`'s `thread: Option<&mut AgentThread>` parameter is unused — a run never writes back to the caller's thread, unlike `ChatAgent::run` |

## Tools & MCP

| Feature | Python | .NET | Rust | Notes |
| --- | --- | --- | --- | --- |
| `Tool` trait / `AiFunction` | ✅ | ✅ | ✅ done | `tools::Tool`, `tools::AiFunction` (closure-backed) |
| Hosted tool markers (code interpreter, web search, file search, MCP) | ✅ (executes server-side) | ✅ | 🚧 partial | `hosted_code_interpreter`/`hosted_web_search`/`hosted_file_search`/`hosted_mcp` are pass-through markers only (`executor: None`) — the provider must support them server-side; there is no local emulation |
| `ApprovalMode` / approval flow | ✅ | ✅ | ✅ done | enforced in `FunctionInvokingChatClient::get_response` |
| MCP client — stdio transport | ✅ | ✅ (via `ModelContextProtocol` SDK integration) | ✅ done | `agent-framework-mcp::McpStdioTool` |
| MCP client — streamable HTTP transport | ✅ | ✅ | ✅ done | `McpStreamableHttpTool` |
| MCP client — WebSocket transport | ✅ | not verified | ❌ not yet | explicitly out of scope per `agent-framework-mcp`'s own doc comment |
| MCP prompts (`prompts/list` / `prompts/get`) | ✅ | not verified | ❌ not yet | only tools are exposed as agent functions |
| MCP sampling / roots callbacks | ✅ | not verified | ❌ not yet | server-initiated requests are logged and ignored, never answered |

## Middleware & memory

| Feature | Python | .NET | Rust | Notes |
| --- | --- | --- | --- | --- |
| Agent middleware pipeline | ✅ | ✅ | ✅ done | `middleware::{Middleware, MiddlewarePipeline, Next}`, wired into `ChatAgent::run`/`run_stream` |
| Chat middleware pipeline | ✅ | ✅ | 🚧 partial | `ChatContext` + the `Middleware<ChatContext>` type alias exist, but nothing currently wires a chat-middleware pipeline into a `ChatClient` call path |
| Function-invocation middleware pipeline | ✅ | ✅ | 🚧 partial | `FunctionInvocationContext` type exists; `FunctionInvokingChatClient` does not yet run calls through it |
| `ContextProvider` / `AggregateContextProvider` | ✅ | ✅ | ✅ done | `memory::ContextProvider`, fan-out/merge over multiple providers |
| Mem0-backed memory provider | ✅ (`mem0` package) | ✅ (`Microsoft.Agents.AI.Mem0`) | ❌ not yet | |
| Redis-backed memory / thread store | ✅ (`redis` package) | not found | ❌ not yet | |

## Workflow engine

| Feature | Python | .NET | Rust | Notes |
| --- | --- | --- | --- | --- |
| Superstep (Pregel/BSP) graph engine | ✅ | ✅ | ✅ done | `workflow::{WorkflowBuilder, Workflow, WorkflowRun}`; `Single`/`FanOut`/`FanIn` edges, switch/case sugar |
| Checkpointing (in-memory + file-backed) | ✅ | ✅ | ✅ done | `CheckpointStorage` trait, `InMemoryCheckpointStorage`, `FileCheckpointStorage` (atomic write via temp+rename); fires automatically every superstep via `with_checkpointing` |
| Checkpoint graph-signature validation on resume | ✅ | ✅ | ❌ not yet | `Workflow::run_from_checkpoint` / `WorkflowRun::restore` trusts the caller: it never checks that the resuming graph's executor ids/edges match what the checkpoint was taken from, and silently skips restoring state for any checkpointed executor id absent from the new graph |
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
| Magentic human-in-the-loop plan review | ✅ (`MagenticHumanInterventionKind.PLAN_REVIEW`) | n/a (no Magentic orchestrator found) | ❌ not yet | the Rust orchestrator goes straight from planning into the round loop with no pause/approve/revise step |
| Workflow-as-agent | ✅ | ✅ | ✅ done | see Agents section |

## Observability

| Feature | Python | .NET | Rust | Notes |
| --- | --- | --- | --- | --- |
| GenAI-semantic-convention tracing spans | ✅ | ✅ | ✅ done | `observability::ObservableChatClient` (`chat` span); `invoke_agent` span in `ChatAgent::run_core`; `execute_tool` span in the function-invocation loop |
| OpenTelemetry SDK exporter wiring | ✅ | ✅ | 🚧 partial | spans follow OTel GenAI conventions and are bridge-ready (e.g. via `tracing-opentelemetry`), but no OTel SDK/exporter is wired up or shipped |

## Ecosystem

| Feature | Python | .NET | Rust | Notes |
| --- | --- | --- | --- | --- |
| A2A (Agent2Agent protocol) | ✅ (`a2a`) | ✅ (`Microsoft.Agents.AI.A2A`, `.Hosting.A2A[.AspNetCore]`) | ❌ not yet | |
| AG-UI protocol | ✅ (`ag-ui`) | ✅ (`Microsoft.Agents.AI.AGUI`) | ❌ not yet | |
| DevUI | ✅ (`devui`) | ✅ (`Microsoft.Agents.AI.DevUI`) | ❌ not yet | |
| Hosting (ASP.NET Core / Azure Functions / Durable Task) | n/a | ✅ (`Microsoft.Agents.AI.Hosting*`, `.DurableTask`) | ❌ not yet | |
| Declarative / YAML agent & workflow definitions | ✅ (`declarative`) | ✅ (`Microsoft.Agents.AI.Declarative`, `.Workflows.Declarative[.AzureAI]`) | ❌ not yet | |
| CopilotStudio integration | ✅ (`copilotstudio`) | ✅ (`Microsoft.Agents.AI.CopilotStudio`) | ❌ not yet | |
| Guardrails (dedicated module) | ❌ (middleware-based, no dedicated module) | ❌ (middleware-based) | ❌ not yet | none of the three implementations ship a first-class guardrails module; all three can express it via their middleware pipeline |
| Other upstream integrations (ChatKit, Purview, Azure AI Search connector, CosmosDB thread store, ...) | ✅ (various packages) | ✅ (various packages) | ❌ not yet | not individually tracked in this matrix |

## Summary of deliberate, known gaps

These are the gaps the maintainers consider worth calling out explicitly (either because they're easy to reach for and surprising when missing, or because they're natural next steps):

- **MCP**: no WebSocket transport, no prompts capability, no sampling/roots callback handling.
- **Magentic**: no human-in-the-loop plan-review pause (Python-only feature; not even present in .NET's Workflows package).
- **WorkflowAgent**: does not update the caller's `AgentThread` history after a run.
- **Hosted tools** (code interpreter, web search, file search, hosted MCP): pass-through markers only — no local emulation, provider must support them server-side.
- **Ecosystem**: no A2A, AG-UI, DevUI, hosting integrations, declarative/YAML definitions, or CopilotStudio.
- **Anthropic**: `ResponseFormat`/structured output is not mapped onto the Messages API.
- **Reliability**: no built-in retry/backoff policy layer for any provider.
- **Checkpointing**: resuming from a checkpoint does not validate that the workflow graph matches the one the checkpoint was taken from.
