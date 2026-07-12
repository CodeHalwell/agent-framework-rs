# Gap analysis: agent-framework-rs vs. the official Agent Framework

A fresh, full-repo audit of this port against the upstream
[Microsoft Agent Framework](https://github.com/microsoft/agent-framework) at
revision `638fbb5f` (2025-12-10) — the same checkout `PARITY.md` was written
against. Unlike `PARITY.md` (the port's own tracking matrix), this document is
an independent re-derivation: six subsystem audits enumerated the upstream
Python and .NET surface file-by-file and checked every feature against the
Rust source, then the highest-impact claims were re-verified by hand against
specific lines.

**Repo health at audit time:** 771 tests pass / 0 fail (`cargo test
--workspace --all-features`), clippy clean, rustfmt clean, CI configured.

**Overall verdict:** the port's breadth is real — every upstream Python
package except `chatkit`, `azurefunctions`, and `lab` has a Rust counterpart,
and the workflow/orchestration/HITL scaffolding is genuinely there. But the
audit found three kinds of problems `PARITY.md` does not tell you about:

1. **Silent-data-loss bugs** in shipped code paths (multimodal input dropped,
   citations discarded, a checkpoint field omitted, hosted tools mis-emitted).
2. **Cross-cutting architectural gaps** (no trait-level streaming, no per-run
   options, hand-written tool schemas, no thread persistence) that cap what
   every downstream crate can do.
3. **Whole missing surfaces that no matrix row tracks** (OpenAI/Azure
   Assistants clients, the Azure Responses client, the AG-UI *client*, the
   DevUI conversations API, agent-level metrics, `as_mcp_server`, …).

Legend: 🐛 bug in shipped code · 🧱 architectural gap · ❌ missing surface ·
🚧 partial. File references are `crate/path:line` for Rust and
`python/packages/...` for upstream.

---

## Status update (post-audit implementation waves)

Four implementation waves landed after this audit; the workspace went from
771 to ~1,100 tests, all green, clippy/rustfmt clean throughout.

**Addressed:**

- **Every item in §1 and §2** except the two called out below: multimodal
  input mapping (1.1), Anthropic betas + hosted tools (1.2), Azure AI tool
  configs + existing-agent merge (1.3), trait-level `run_stream` with real
  SSE through hosting/AG-UI/orchestrations (1.4), per-run `AgentRunOptions`
  + declaration-only tool round-trips (1.5), `AiFunction::typed` schema
  derivation via schemars (1.6), thread serialize/deserialize + store
  factory (1.7 — container keys match Python; message payloads still
  serialize in Rust's shape), fan-in checkpoint capture (1.8), concurrent
  within-superstep execution (1.9), and all fifteen §2 items (citations,
  reasoning/hosted outputs, usage breakdowns, `thread_created`,
  `.value` auto-fill, `#[serde(other)]`, name sanitization, …).
- From §1.10: the **OpenAI Assistants client** and the **Azure OpenAI
  Responses client** now exist. Still open from that item: an
  `AzureOpenAIAssistantsClient` convenience wrapper and the new Foundry
  Prompt-Agent client.
- From §3: granular errors (`ServiceInvalidAuth` / `ServiceInvalidRequest` /
  `ServiceContentFilter` + provider classification, retry-excluded),
  observability request/tool span attributes + `gen_ai.provider.name`, and
  GenAI **metrics** (token-usage / operation-duration / function-invocation
  histograms) behind an `otel-metrics` feature; `invoked(error)` context-
  provider observability; per-function invocation limits; hosted-tool config
  setters; `DefaultAzureCredential` / `EnvironmentCredential` /
  `WorkloadIdentityCredential`.
- From §4: MCP servers as **first-class agent tools** (`ToolSource`,
  resolved per run) with `list_changed` invalidation, load flags,
  stdio/websocket request timeouts, and name dedup.
- A top-level `examples/` gallery crate: 69 categorized examples (the 32
  originals moved + 37 new, including showcases for each new capability),
  indexed in `examples/README.md`.

**Still open** (the honest remainder): typed executor routing /
`AgentExecutorRequest`-style envelopes / sub-workflow request interception
and the other §4 workflow-depth items; orchestration builder HITL/
checkpointing options; cross-language wire `type` tags (and
`raw_representation` / `additional_properties` on content types); richer
middleware contexts; `as_mcp_server`; AG-UI client + predictive-state
events; the DevUI conversations/cancel/discovery/auth surface; A2A serving
streaming + task lifecycle; the Foundry Prompt-Agent client; OTel exporter
wiring (by design — the app installs the SDK); and the big tracked
ecosystem items (DurableTask/Azure Functions hosting, ChatKit, Copilot
Studio workflow DSL, Redis vector/hybrid retrieval, Cosmos hardening +
checkpoint store, Purview protection scopes, `lab`).

The sections below are preserved as the audit's original findings; read
them together with this status block.

---

## 1. The headline items

Ranked by impact on someone trying to use this port for real work today.

### 1.1 🐛 Multimodal input is silently dropped (OpenAI Chat + Responses)

`messages_to_openai` matches only `Text`, `FunctionCall`, `FunctionResult`
and discards every other content variant via a `_ => {}` arm
(`agent-framework-openai/src/convert.rs:20-26`); the Responses-API input
mapper does the same (`responses.rs:~312`). Upstream maps `DataContent`/
`UriContent` to `image_url`, `input_audio`, and `file` parts
(`openai/_chat_client.py:407-449`, `_responses_client.py:543-582`). A user
attaching an image gets no error — the model simply never sees it. The same
`_ => {}` pattern means Azure OpenAI (which reuses this converter) is equally
affected.

### 1.2 🐛 Anthropic: beta flags never sent, hosted tools silently filtered

The Python client always enables `mcp-client-2025-04-04` and
`code-execution-2025-08-25` betas and maps hosted web-search /
code-execution / MCP-connector tools (`anthropic/_chat_client.py:54,
254-264, 397-421`). The Rust client sends no `anthropic-beta` header at all
(verified: zero hits in the crate) and filters the tool list to
`ToolKind::Function` (`agent-framework-anthropic/src/convert.rs:305-317`),
so hosted tools vanish without error. Hosted-tool result blocks
(`server_tool_use`, `mcp_tool_result`, `web_search_tool_result`, …) and
citations are likewise never parsed. The default `max_tokens` also diverges
(Rust 4096 vs Python 1024).

### 1.3 🐛 Azure AI Foundry hosted tools are emitted in a non-functional form

`bing_grounding` is emitted without its required `connection_id` (the service
rejects it) and `file_search` without `vector_store_ids`/`tool_resources`
(it searches nothing) — `agent-framework-azure-ai/src/convert.rs:176-177` vs
`azure-ai/.../_chat_client.py:924-998`. Also missing: existing-agent
definition merge (tools/instructions of an `agent_id` are not fetched), Bing
Custom Search, and MCP `tool_resources` headers.

### 1.4 🧱 `run_stream` is not on the `Agent` trait

The object-safe trait has only `run`
(`agent-framework-core/src/agent.rs:29-56`); `run_stream` is an inherent
`ChatAgent` method. Upstream puts streaming on the agent protocol itself
(`_agents.py:214-238`). Consequences cascade everywhere a `dyn Agent` is
held:

- All three hosting routers run the agent to completion and replay the result
  as fake SSE (`agent-framework-hosting/src/{devui/mod.rs:195, agui.rs:208,
  openai_compat.rs:97}`).
- Workflows/orchestrations emit `AgentRunUpdate` post-hoc per final message —
  no token-level `AgentRunUpdateEvent` streaming inside workflows
  (`orchestration/mod.rs:105-133` vs `_agent_executor.py:268-360`).
- `CopilotStudioAgent::run_stream` and A2A-agent streaming can't exist behind
  the trait.

Fixing this one trait (e.g. a `run_stream` returning
`Pin<Box<dyn Stream<...>>>` with a default buffered impl) unblocks streaming
across hosting, workflows, and remote agents at once.

### 1.5 🧱 No per-run option overrides

Upstream `run`/`run_stream` accept the full option set per call (temperature,
tools, tool_choice, model, response_format, …) merged over agent defaults
(`_agents.py:770-1046`; .NET `AgentRunOptions`). Rust's `Agent::run(messages,
thread)` takes nothing else — changing anything means rebuilding the agent.
This also blocks: AG-UI client-declared tools injection (parsed then dropped,
`hosting/src/agui.rs:422-423`), `as_tool` runtime-kwarg forwarding, and
run-level middleware.

### 1.6 🧱 Tool schemas are hand-written

Python's headline `ai_function` derives the JSON-Schema parameters from type
hints (`_tools.py:903-1103`); .NET does the reflection equivalent. Rust's
`AiFunction::new` takes a raw `serde_json::Value` schema
(`core/src/tools.rs:170-187`) — `schemars` is a declared dependency but
unused for tools, and there is no derive/proc-macro. This is the port's
single biggest ergonomic regression.

### 1.7 🧱 No thread persistence (serialize/deserialize/resume)

Upstream round-trips conversations: `AgentThread.serialize()`,
`AgentThread.deserialize()`, `BaseAgent.deserialize_thread`
(`_threads.py:421-506`, `_agents.py:378`), and .NET
`AgentThread.Serialize`/`AIAgent.DeserializeThread`. Rust's `AgentThread`
has no serialize/deserialize at all (`core/src/threads.rs`), local threads
are always `InMemoryChatMessageStore` (no `chat_message_store_factory`
equivalent, `agent.rs:527-541`), and `ChatMessageStore` has no
`deserialize`/`update_from_state`. Related: the serialized wire shape of
core types diverges from upstream (no `type` discriminators; `Role`/
`FinishReason` as bare strings), so even hand-rolled persistence is not
cross-language compatible with Python/.NET stores.

### 1.8 🐛 Workflow checkpoints lose partial fan-in state

The runner's fan-in buffer (`workflow/runner.rs:507`, filled at :658,
cleared only on completion at :671) is not a field of `WorkflowCheckpoint`
(`workflow/checkpoint.rs:34-56`) and `maybe_checkpoint` never captures it
(`runner.rs:774-800`). A checkpoint taken while a fan-in is partially
satisfied across supersteps silently loses the already-delivered messages on
resume — the fan-in can then never fire. Upstream checkpoints carry edge/
fan-in runner state (`_runner_context.py:459-475`; .NET
`Checkpointing/FanInEdgeState.cs`).

### 1.9 🐛 "Concurrent" execution is sequential

Within a superstep the Rust runner awaits each executor one after another in
sorted order (`workflow/runner.rs:626-693`, verified); upstream delivers
concurrently via `asyncio.gather` (`_runner.py:177-182`). Since executors
here are usually LLM calls, `ConcurrentBuilder` with N agents currently runs
~N× slower than upstream's.

### 1.10 ❌ Assistants-family clients don't exist

`OpenAIAssistantsClient` (`openai/_assistants_client.py`, 538 lines),
`AzureOpenAIAssistantsClient`, and `AzureOpenAIResponsesClient`
(`azure/_{assistants,responses}_client.py`) ship in both Python and .NET.
None exist in Rust (grep for assistants/threads/runs surface: zero hits in
the openai/azure crates), and `PARITY.md` never mentions them — for Azure
OpenAI, Rust ships Chat Completions only. Also absent: the newer
`AzureAIClient` Foundry Responses client (`azure-ai/_client.py`) and .NET's
new Foundry "Prompt Agent" client (versioned server-side agents) — the Rust
`agent-framework-azure-ai` crate corresponds only to the older
Persistent/Assistants flavor.

---

## 2. Silent data-loss & correctness list (all verified against source)

| # | Issue | Where | Upstream behavior |
|---|---|---|---|
| 1 | Image/audio/file input dropped | `openai/src/convert.rs:25`, `responses.rs:312` (and Azure OpenAI via reuse) | mapped to `image_url`/`input_audio`/`file` parts |
| 2 | Annotations/citations never parsed back (url/file citations) | `openai/src/responses.rs:456-513`; same for Anthropic (`convert.rs:373-412`) | folded into `TextContent.annotations` (`_responses_client.py:667-724`) — only the azure-ai SSE parser does this in Rust (`sse.rs:198-238`) |
| 3 | Anthropic betas + hosted tools dropped | `anthropic/src/{lib.rs:150-174, convert.rs:305}` | always-on betas; hosted tools mapped |
| 4 | Hosted tools mis-emitted as `{"type":"function"}` on Chat Completions; `HostedWebSearchTool` not mapped to `web_search_options` | `core/src/tools.rs:121`, `openai/src/convert.rs:99-113` | dedicated mappings (`_chat_client.py:140-175`) |
| 5 | `logit_bias`, `metadata`, `parallel_tool_calls` never sent (Chat Completions) | `openai/src/convert.rs:116-143` | sent |
| 6 | Azure AI `bing_grounding`/`file_search` emitted without connection/vector-store config | `azure-ai/src/convert.rs:176-177` | required params attached |
| 7 | Fan-in buffer missing from checkpoints | `workflow/runner.rs:774-800` vs `:507,658-671` | runner state checkpointed |
| 8 | Sequential within-superstep execution | `workflow/runner.rs:626-693` | `asyncio.gather` |
| 9 | `ContextProvider::thread_created` never invoked by `ChatAgent` — dead hook (mem0/redis implement it; nothing calls it) | `core/src/agent.rs` (zero call sites) | called on thread creation/service-thread adoption (`_agents.py:1228-1265`) |
| 10 | `ChatResponse.value` never auto-populated when `response_format` is set (manual `parse_json` only) | `core/src/types/response.rs:55` (no assignment anywhere) | `try_parse_value` fills it (`_types.py:2551-2557`) |
| 11 | Unknown content `type` fails whole-message deserialization (closed enum, no `#[serde(other)]`) | `core/src/types/content.rs:305-318` | logged and skipped (`_types.py:2205-2210`) |
| 12 | OpenAI Responses: reasoning stream events, `code_interpreter_call` outputs, `image_generation_call`, MCP approval round-trip all dropped; `ToolKind` cannot even represent image-generation/computer-use | `responses.rs:456-513,659-798`; `core/src/tools.rs:37` | parsed / representable |
| 13 | Usage detail breakdown (reasoning/audio/cached token counts) dropped | `openai/src/convert.rs:212-219` | mapped into `UsageDetails` extras |
| 14 | Service thread returning no conversation id is silently ignored | `core/src/threads.rs:128-150` | raises `AgentExecutionException` |
| 15 | `as_tool` uses the raw agent name as the function name (no sanitization → can emit invalid tool names) | `core/src/agent.rs:706-742` | `_sanitize_agent_name` (`_agents.py:53-87`) |

## 3. Cross-cutting architectural gaps

| Gap | Rust today | Upstream | Impact |
|---|---|---|---|
| Streaming on the agent abstraction | inherent `ChatAgent::run_stream` only | protocol-level `run_stream` | fake SSE in hosting; no in-workflow token streaming (§1.4) |
| Per-run options / run-level middleware | `run(messages, thread)` | full kwargs merge; `AgentRunOptions`; `middleware=` per call | §1.5 |
| Tool schema derivation | hand-written `Value` | type-hint/reflection derived | §1.6 |
| Thread + store persistence | none | serialize/deserialize/resume + store factory | §1.7 |
| Typed executor routing in workflows | single untyped `execute(Value)` | `@handler` multi-dispatch by type, `@executor`, `@response_handler`, input/output type introspection | biggest conceptual divergence in the engine; also blocks type-compat validation |
| Cross-language wire format | serde-native, no `type` tags | discriminated `to_dict` shape | Rust-persisted threads/checkpoints unreadable by Python/.NET (and vice versa); Redis provider additionally stores JSON where Python uses hashes → shared index incompatible |
| Exceptions | flat 12-variant `Error` | 17-class hierarchy | no distinct content-filter or invalid-auth errors to branch/retry on |
| Middleware contexts | no `agent`/`thread`/function-object/`chat_client`/kwargs exposure; no unified list/decorators | rich contexts, unified registration | limits real middleware (guardrails, routing) |
| Observability | spans only, older `gen_ai.system` attr, small attribute set | + token-usage & duration **histograms**, `setup_observability()`, OTLP/App-Insights env wiring, richer span attrs | no metrics at all today; PARITY's 🚧 row undersells this |
| `Content` field parity | no `additional_properties`/`raw_representation` on any content/message/response/update | on everything (`BaseContent`) | provider metadata lost, esp. across streaming aggregation |

## 4. Missing surfaces not tracked by PARITY.md

- **Providers:** `OpenAIAssistantsClient`, `AzureOpenAIResponsesClient`,
  `AzureOpenAIAssistantsClient`, `AzureAIClient` (Foundry Responses), .NET
  Foundry Prompt-Agent client; `DefaultAzureCredential` /
  `EnvironmentCredential` / `WorkloadIdentityCredential` (the managed-cloud
  defaults; Rust requires hand-assembling `ChainedTokenCredential`).
- **Agents:** `as_mcp_server` (expose an agent as an MCP server,
  `_agents.py:1095-1202`); `chat_message_store_factory`;
  `get_new_thread(service_thread_id=…)`; agent-level per-function invocation
  limits (`max_invocations`).
- **MCP:** MCP tools as first-class agent tools (auto-connect at run time —
  Rust requires manual `tool_definitions().await` wiring, tools frozen at
  build); `notifications/tools|prompts/list_changed` reload; `HostedMCPTool`
  approval-mode dict/headers/description.
- **AG-UI:** the entire **client** (`AGUIChatClient` + converters,
  `ag-ui/.../_client.py`, 407 lines) — consume a remote AG-UI agent as a
  `ChatClient`; predictive-state events (`STATE_SNAPSHOT`/`STATE_DELTA`
  RFC-6902, `MESSAGES_SNAPSHOT`, `PredictState`, `confirm_changes` HITL +
  confirmation strategies) which **both** Python and .NET ship; state-schema
  context injection.
- **DevUI parity:** Rust implements 4 of the Python server's 21 routes. Absent:
  Conversations + items API (10 routes, incl. workflow checkpoint-resume),
  run cancellation (`POST /v1/responses/{id}/cancel` + disconnect-driven),
  `GET /meta`, directory-based entity discovery + hot reload
  (`devui ./agents`), bearer-token auth, CORS, OpenAI proxy mode, inline
  OTel trace events, Azure Container Apps deployment endpoints.
- **A2A serving:** `message/stream` is rejected (`UNSUPPORTED_OPERATION`,
  card advertises `streaming:false`), tasks are terminal-only (no
  working/input-required lifecycle), inbound file/data parts are dropped
  (text-only), no persistent thread store keyed by `contextId` (.NET:
  `AgentThreadStore`), plus the already-tracked push-config/resubscribe/
  extended-card trio (`hosting/src/a2a.rs:280-388`).
- **Workflows:** `AgentExecutorRequest/Response` envelopes + `should_respond`;
  tool-approval → `request_info` bridging inside workflows; builder
  `add_agent`/`register_agent`; parent interception of sub-workflow requests
  (`SubWorkflowRequestMessage.create_response`); orchestration
  `with_checkpointing` / `with_request_info` HITL / Concurrent custom
  aggregator / mixed executor participants / group-chat human participants;
  streaming resume (`send_responses_streaming`); event-origin +
  `WorkflowWarningEvent` + structured `WorkflowErrorDetails`; self-loop &
  dead-end validation; viz file export + nested sub-workflow rendering;
  `CosmosCheckpointStore` equivalent (durable checkpoint backend).
- **Memory/context:** `invoked(invoke_exception=…)`; provider async
  lifecycle; in-box `ChatHistoryMemoryProvider`/`TextSearchProvider` (.NET);
  Mem0 local/OSS mode + graph `relations`; Redis vector/KNN + hybrid search
  (tracked) *and* hash-vs-JSON wire compatibility (untracked).
- **Hosting runtime (.NET-only but the production story):** DI agent
  registry + `IAgentThreadStore` server-side thread persistence, stateful
  OpenAI **Responses + Conversations** serving (get/cancel/input_items,
  conversation CRUD), `WorkflowCatalog`.

## 5. Tracked-and-confirmed gaps (PARITY.md roadmap is accurate here)

Durable execution (DurableTask / `azurefunctions` durable entities,
orchestration triggers, MCP triggers), ChatKit, `lab` (gaia / lightning /
tau2 eval+RL tooling), MCP elicitation + GET-SSE listening + auto-reconnect,
Redis vector/hybrid retrieval, Cosmos chat-store hardening (Entra auth,
TransactionalBatch, hierarchical PK, TTL), Purview `ScopedContentProcessor`
(protection-scopes precheck/caching, offline mode, background audit, JWT
identity), the Copilot-Studio declarative workflow DSL (~24k lines of .NET:
PowerFx interpreter + ~20 action kinds + codegen — plus untracked: declarative
agent PowerFx `=Env.X` expressions and prompt `template` Format/Parser),
Azure AI Search agentic Knowledge-Base mode, the React DevUI, and OTel
exporter wiring. Also verified accurate: the A2A client (exceeds Python),
Copilot Studio conversation continuity (exceeds Python), Anthropic
structured-output folding (exceeds both), Magentic plan-review /
stall-intervention HITL, checkpoint graph-signature validation,
`WorkflowRunState` (all 7 states), and the retry layer.

## 6. PARITY.md rows needing a status correction

| Row (PARITY.md) | Claimed | Should be | Why |
|---|---|---|---|
| OpenAI Chat Completions (L34) | ✅ | 🚧 | multimodal dropped; logit_bias/metadata/parallel_tool_calls unsent; hosted tools mis-emitted |
| OpenAI Responses (L35) | ✅ | 🚧 | multimodal in + citations out + hosted outputs + reasoning stream dropped |
| Azure AI Foundry (L38) | ✅ | 🚧 | bing/file_search non-functional as emitted; agent-def merge absent; conflates Persistent vs new Foundry client |
| Anthropic (L39) | ✅ | 🚧 | no beta headers; hosted tools filtered; citations dropped |
| Hosted tool markers (L62) | 🚧 "pass-through" | 🚧 (reword) | not passed through on Chat Completions (mis-emitted as function) or Anthropic (dropped) |
| `AgentThread`/store (L50-51) | ✅ | 🚧 | no serialize/deserialize/resume |
| `agent.as_tool()` (L52) | ✅ | 🚧 | no stream_callback/kwargs/name sanitization |
| `ContextProvider` (L79) | ✅ | 🚧 | `thread_created` never fired; no `invoke_exception` |
| `Tool`/`AiFunction` (L61) | ✅ | 🚧 | no schema derivation from types |
| Middleware pipelines (L76-78) | ✅ | 🚧 | impoverished contexts; no unified/run-level registration |
| Superstep engine (L89) | ✅ | 🚧 | sequential within-superstep |
| Checkpointing (L90) | ✅ | 🚧 | fan-in buffer lost; no streaming resume |
| Request/response HITL (L92) | ✅ | 🚧 | untyped; no `send_responses_streaming`; no agent-approval bridging |
| Graph validation (L94) | ✅ | 🚧 | self-loop/dead-end checks absent (structural, not type-based) |
| Visualization (L95) | ✅ | 🚧 | no file export; no nested sub-workflow rendering |
| Sub-workflows (L96) | ✅ | 🚧 | parent request interception absent (functional, not "shape") |
| Sequential/Concurrent/Group chat (L102-105) | ✅ | 🚧 | agents-only; no custom aggregator/HITL/checkpointing; group chat single-executor |
| Content union / ChatResponse / ChatOptions rows (L17-26) | ✅ | 🚧 (annotate) | missing additional_properties/raw_representation; `.value` not auto-filled; options not serializable/validated; wire shape diverges |
| GenAI tracing spans (L116) | ✅ | 🚧 | old `gen_ai.system` attr; small attribute subset; **no metrics** (L117's note doesn't mention metrics at all) |
| DevUI-style API (L126) | ✅ | 🚧 | 4/21 routes |
| OpenAI-compatible serving (L128) | ✅ "(via devui)" | ✅ (fix attribution) | Python devui serves Responses/Conversations, not `/v1/chat/completions` (that's .NET Hosting.OpenAI) |
| AG-UI (L132) | ✅ | 🚧 | client entirely absent; predictive-state/confirm_changes missing (shipped by both upstreams) |
| Mem0 (L80) | ✅ | ✅ (annotate) | hosted-only; no graph relations |
| Cosmos (L83) | 🚧 | 🚧 (extend) | also missing: `CosmosCheckpointStore` workflow-checkpoint backend |

## 7. Suggested priority order

1. **Fix the silent-loss bugs** (§2 items 1-8) — small diffs, immediate
   correctness value: multimodal mapping, citation parsing, Anthropic betas +
   hosted tools, Azure AI tool params, checkpoint fan-in field, concurrent
   superstep delivery (`futures::future::join_all`), `thread_created` call,
   `#[serde(other)]` unknown-content fallback.
2. **Trait-level streaming + per-run options** (§1.4, §1.5) — two API changes
   that unblock hosting SSE, in-workflow streaming, AG-UI tools injection,
   and run-level middleware everywhere.
3. **`#[derive]`/macro tool schemas** via the already-present `schemars` dep.
4. **Thread serialize/deserialize + store factory**, ideally with the
   upstream-compatible `type`-tagged wire shape (or a compat serializer).
5. **Assistants/Responses client family** (OpenAI + Azure OpenAI) and
   `DefaultAzureCredential`.
6. **Workflow depth**: AgentExecutor envelopes + approval bridging, builder
   `add_agent`, orchestration HITL/checkpointing options, sub-workflow
   interception, event-model completion.
7. **Ecosystem depth** in whatever order matters to your users: DevUI
   conversations/cancel/discovery, AG-UI client + state events, A2A serving
   streaming/lifecycle, metrics + `setup_observability`, then the big tracked
   items (durable hosting, ChatKit, declarative DSL, Redis vectors).

---

*Method note: six parallel subsystem audits (core types/clients; agents,
tools, MCP, memory, middleware; workflow engine; providers; ecosystem
packages; .NET-only surface) each enumerated upstream source directly and
grepped/read this repo for equivalents; the synthesis re-verified the
highest-impact claims (fan-in checkpoint omission, sequential superstep,
multimodal drop, Anthropic betas, hosting run-to-completion, Agent trait
surface) line-by-line before inclusion.*
