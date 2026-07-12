# Upstream drift: re-baselining agent-framework-rs against current upstream

This document catalogs **everything that must change across the whole library**
to bring this Rust port back in line with the current
[Microsoft Agent Framework](https://github.com/microsoft/agent-framework).

## Why this document exists

The port — and `PARITY.md` and `GAP_ANALYSIS.md` with it — was built against
upstream revision **`638fbb5f`** (2025-12-10). That checkout is now roughly
**seven months stale**. Current upstream `main` is **`68136ee`** (2026-07).
In that window upstream landed the largest set of breaking changes in the
framework's history: the core package alone changed **+43,484 / −23,767 lines
across 122 files**, providers were extracted out of core into standalone
packages, orchestrations were extracted into their own package, threads became
sessions, and — the change that prompted this re-baseline — **`ChatAgent` was
renamed to `Agent`.**

That last point matters beyond the rename itself. The original question
"why is it `ChatAgent` and not `Agent`, which is what Python uses?" was
**correct**, and the port answering "Python also uses `ChatAgent`" was wrong:
it was reading a checkout from before upstream renamed the class. The stale
baseline invalidated conclusions throughout the earlier audit. The clearest
example: the OpenAI **Assistants** client the port built in Wave 3 and then
polished across four PR-review rounds targets an API that upstream has since
**removed entirely**, and the "missing Foundry Prompt-Agent client" that the
gap analysis flagged as a hole is now the shipping `foundry` package.

### How this was derived

Six parallel subsystem investigations diffed `638fbb5f..68136ee`
file-by-file and mapped each change onto the Rust source in
`crates/agent-framework-core/` and the provider/ecosystem crates:

1. Core agents / sessions / memory / chat client
2. Core types / tools / middleware / observability / new modules
3. Workflow engine (builder, graph, edges, validation, viz, checkpoint, runner, state)
4. Orchestrations (Sequential, Concurrent, GroupChat, Handoff, Magentic)
5. Providers (OpenAI, Azure, Foundry, Anthropic, and new provider packages)
6. Hosting & ecosystem packages

Every claim below cites upstream Python (`python/packages/...:line`) and, where
an action is required, the Rust site (`crate/path.rs:line`). Line numbers are
from the two pinned revisions and will drift as files are edited.

### Relationship to the other docs

- **`PARITY.md`** — the port's own tracking matrix, at `638fbb5f`. Its rows are
  now baseline-stale; treat this document as the authority on what upstream
  looks like today.
- **`GAP_ANALYSIS.md`** — the independent audit, also at `638fbb5f`. It is
  corrected separately (see the companion re-baseline pass); this document is
  the input to that correction.

### Change classification

| Tag | Meaning |
|---|---|
| **[RENAME]** | Symbol renamed, role unchanged. |
| **[SIGNATURE]** | Parameters / return type / fields changed. |
| **[MOVED]** | Relocated to another module or package (possibly reshaped). |
| **[REMOVED]** | Deleted from the public surface, no direct replacement. |
| **[NEW]** | New capability with no `638fbb5f` equivalent. |
| **[KEPT]** | Verified unchanged (called out to prevent needless churn). |
| **[BEHAVIOR]** | Runtime behavior changed even if the signature did not. |

---

## Part I — Cross-cutting architectural themes

Six themes recur across every subsystem. Understanding them first makes the
per-file actions in Part II read as instances of a handful of moves rather than
a hundred unrelated edits.

### Theme A — The `Agent` rename (protocol vacates the name)

Upstream renamed the run **protocol** to free the bare name `Agent` for the
concrete class:

| `638fbb5f` | `68136ee` | |
|---|---|---|
| `AgentProtocol` (Protocol) | `SupportsAgentRun` (Protocol) | [RENAME] |
| `ChatAgent(BaseAgent)` | `Agent(AgentMiddlewareLayer, AgentTelemetryLayer, RawAgent)` | [RENAME] + layers |
| — | `RawAgent(BaseAgent)` | [NEW] minimal base |

`Agent` is now the recommended class; `RawAgent` is the middleware/telemetry-free
core (`_agents.py:663`), and `SupportsAgentRun` is the structural protocol
(`_agents.py:213`). This is the single largest mechanical change for the port:
in Rust, `trait Agent` currently occupies the name that `struct ChatAgent` must
take, so the rename must happen as a pair — `trait Agent`→`SupportsAgentRun`
first, then `struct ChatAgent`→`Agent`. It ripples through **~40 files**: every
`Arc<dyn Agent>` in orchestration, both `impl Agent for …`, and all prelude /
umbrella re-exports.

### Theme B — Unified `run(stream=)` / `get_response(stream=)`

The separate streaming entry points were folded into a single overloaded call
that takes a `stream: bool`:

- `Agent.run_stream(...)` → `Agent.run(..., stream=True)` (`_agents.py:1749`).
- `ChatClient.get_streaming_response(...)` →
  `get_response(..., stream=True)` (`_clients.py:482`).
- `BaseChatClient` collapsed two abstract methods (`_inner_get_response` +
  `_inner_get_streaming_response`) into one `_inner_get_response(*, stream)`.
- `Workflow.run_stream()` / `send_responses()` / `send_responses_streaming()` →
  one `Workflow.run(message=None, *, stream=False, responses=None, …)`
  (`_workflow.py:668`).

Streaming now returns a first-class `ResponseStream[Update, Final]`
(`_types.py:3047`) — an async iterable of updates that also yields a final
aggregated result. Rust currently exposes paired methods and returns update
vecs/streams; every client, agent, and workflow entry point changes shape.

### Theme C — Threads → Sessions, and Memory merged in

`_threads.py` and `_memory.py` were **both deleted** and their replacements
consolidated into a new `_sessions.py` (1308 lines):

- `AgentThread` → **`AgentSession`** — a lightweight `{session_id,
  service_session_id, state: dict}` container. Message storage **left the
  thread**: there is no `message_store` field.
- `ChatMessageStore` / `ChatMessageStoreProtocol` → **`HistoryProvider`**
  (a `ContextProvider`), with `InMemoryHistoryProvider` and new
  `FileHistoryProvider`.
- `Context` → **`SessionContext`** (much richer per-invocation object).
- `ContextProvider.invoking/invoked` → **`before_run`/`after_run`** (now mutate
  the passed `SessionContext` in place instead of returning a `Context`);
  `thread_created` removed.
- **`AggregateContextProvider` deleted** — the agent iterates
  `Sequence[ContextProvider]` directly, attributing messages by `source_id`.
- Service identifier `conversation_id: str` → `ServiceSessionId = Mapping[str, Any]`.
- `serialize`/`deserialize` (async) → `to_dict`/`from_dict` (**sync**).

This is the keystone change: skills, compaction, security, and the harness all
build on the new `SessionContext` / `ContextProvider` shape.

### Theme D — Decorators → composable "Layer" mixins

Four separate "decorator-wraps-object" mechanisms became composable mixin
classes sharing one unified entry point:

| `638fbb5f` decorator | `68136ee` layer |
|---|---|
| `use_function_invocation` | `FunctionInvocationLayer` |
| `use_agent_middleware` / `use_chat_middleware` | `AgentMiddlewareLayer` / `ChatMiddlewareLayer` |
| `use_observability` / `use_agent_observability` | `ChatTelemetryLayer` / `AgentTelemetryLayer` / `EmbeddingTelemetryLayer` |

The MRO-based Layer mechanism is Python-specific; Rust's existing
wrapper-client / pipeline approach (`FunctionInvokingChatClient`,
`MiddlewarePipeline`, `ObservableChatClient`) is an acceptable equivalent and
does **not** need to be rebuilt. What must follow is the *unified signature*
(Theme B) and the new params those layers thread through
(`compaction_strategy`, `tokenizer`).

### Theme E — Providers extracted from core into standalone packages

At `638fbb5f`, the OpenAI, Azure-OpenAI, and Anthropic clients lived **inside**
core (`agent_framework/{openai,azure,anthropic}/`); only `anthropic`,
`azure-ai`, and `copilotstudio` were standalone. At `68136ee`, **every** provider
is a standalone pip package under `python/packages/`, and core keeps only lazy
re-export namespace shims. Nine provider packages exist where three did, and
`azure-ai` was deleted and superseded by `foundry`. The Rust workspace already
has provider crates (so it matches the *structure*), but the crate contents and
names must follow the client renames and additions in §13.

### Theme F — The type system was rebuilt

`_types.py` was rewritten around: a single `Content` class discriminated by a
`type` field (was ~13 `BaseContent` subclasses); `TypedDict` option bags
(`ChatOptions`, `ToolMode`, `UsageDetails`) instead of `SerializationMixin`
classes; `NewType(str)` open enums for `Role`/`FinishReason`; and generic,
structured-output-aware responses. The load-bearing renames that thread through
**every** signature in the library:

- `ChatMessage` → **`Message`**
- `AgentRunResponse` → **`AgentResponse`**, `AgentRunResponseUpdate` → **`AgentResponseUpdate`**
- `ChatResponse.model_id` → `model`; `ChatOptions.model_id` → `model`
- `AIFunction` → **`FunctionTool`**; `ai_function` decorator → `tool`
- `AgentRunContext` → **`AgentContext`**
- `CitationAnnotation` → **`Annotation`**

Rust's design anticipated some of this well: `Content` is already a tagged
enum (keep it — do **not** flatten to a fat struct), `Role`/`FinishReason` are
already open string newtypes, and `ChatResponse.value` already stands in for
generic structured output. The renames and new fields, however, are unavoidable.

---

## Part II — Subsystem-by-subsystem drift and Rust actions

### §1 — Core agents (`_agents.py` → `agent.rs`)

`_agents.py` grew 1307 → 1844 lines. Beyond Theme A/B:

- **[SIGNATURE]** `Agent.__init__` vs old `ChatAgent.__init__`:
  - `chat_client` → **`client`**.
  - ~20 flat chat-option kwargs (`temperature`, `max_tokens`, `tool_choice`, …)
    collapsed into one **`default_options`** TypedDict; per-run override via
    `options=` on `run` (`_agents.py:1781`).
  - `chat_message_store_factory` → **removed** (history is now a `HistoryProvider`).
  - New params **`compaction_strategy`**, **`tokenizer`**,
    **`require_per_service_call_history_persistence`**.
  - `context_providers`: `ContextProvider | list | AggregateContextProvider`
    → `Sequence[ContextProvider]`.
- **[REMOVED]** `get_new_thread` → **`create_session`/`get_session`**
  (`_agents.py:336`).
- **[REMOVED]** `create_agent` factory **deleted entirely** (was
  `BaseChatClient.create_agent(...) -> ChatAgent`). Construct `Agent(client=…)`
  directly.
- **[REMOVED]** `display_name` dropped from the run protocol.

**Rust actions (`crates/agent-framework-core/src/agent.rs`):**
- Rename `trait Agent`→`trait SupportsAgentRun` (`agent.rs:217`) and
  `struct ChatAgent`→`struct Agent` (`agent.rs:300`); `ChatAgentBuilder`→
  `AgentBuilder` (`agent.rs:989`). Do the trait rename first to vacate the name.
- Repoint every `dyn Agent` / `impl Agent`: `impl Agent for ChatAgent`
  (`agent.rs:898`), `impl Agent for WorkflowAgent`
  (`workflow/orchestration/workflow_agent.rs:278`), and all `Arc<dyn Agent>` in
  orchestration — `group_chat.rs` (≥9 sites: `:259,321,429,450,478,535`),
  `concurrent.rs:77,88,95`, `sequential.rs:16,27,34`,
  `handoff.rs:303,512,544`, `magentic.rs:458,872,1334`, `mod.rs:113,154,161`,
  plus the `orchestration_*.rs` tests. Update re-exports: prelude
  (`lib.rs:50-51`), core-lib doc lines, umbrella `agent-framework/src/lib.rs:33`.
- Fold `run_stream` into `run(stream: bool)` (trait methods `agent.rs:222,236,260,263`;
  `ChatAgent::run_stream*` `:379-395`).
- Rename `get_new_thread`→`create_session`/`get_session` (`agent.rs:289`,
  `get_new_thread_with_service_id`/`deserialize_thread` `:1168,1186`).
- Builder: `chat_client`→`client`, drop `chat_message_store_factory`
  (`agent.rs:1087`), collapse flat options into `default_options`, add
  `compaction_strategy`/`tokenizer`/`require_per_service_call_history_persistence`.
- Confirm no `create_agent` shim is added.

### §2 — Sessions (`_threads.py` removed, `_sessions.py` new → `threads.rs`)

See Theme C for the shape. Concretely:

- **[RENAME+RESHAPE]** `AgentThread`→`AgentSession` (`_sessions.py:913`);
  fields `session_id` (property), `service_session_id`, `state: dict`; **no
  `message_store`**.
- **[SIGNATURE]** `serialize`/`deserialize` → sync `to_dict`/`from_dict`
  (`_sessions.py:961,975`).
- **[REMOVED→REPLACED]** `ChatMessageStore(Protocol)` → `HistoryProvider`
  (`_sessions.py:426`), `InMemoryHistoryProvider` (`:988`), new
  `FileHistoryProvider` (`:1068`). Subclasses implement
  `get_messages()`/`save_messages()`; behavior flagged by
  `load_messages`/`store_inputs`/`store_context_messages`/`store_context_from`/
  `store_outputs`.
- **[NEW]** internal middleware `MessageInjectionMiddleware` (`:602`),
  `PerServiceCallHistoryPersistingMiddleware` (`:737`).

**Rust actions:**
- `threads.rs` → conceptually `sessions.rs`: rename `AgentThread`→`AgentSession`
  (`threads.rs:111`), strip `message_store` (`:113`). Rename async
  `serialize`/`deserialize` (`:270,291`) to sync `to_dict`/`from_dict`.
- Move `trait ChatMessageStore` (`threads.rs:26`) + `InMemoryChatMessageStore`
  (`:61`) onto the provider model → `HistoryProvider` / `InMemoryHistoryProvider`,
  add `FileHistoryProvider`. Rework `message_store`/`ensure_local_store`/
  `on_new_messages`/`list_messages` (`:202-241`) so history flows through
  providers writing `session.state`.
- Rename `service_thread_id`→`service_session_id` (`:154,159`) and the
  `try_adopt_service_thread_id` logic (`:177`) to the map-typed `ServiceSessionId`.
- Update prelude re-export (`lib.rs:64`).

### §3 — Memory / context providers (`_memory.py` removed → `memory.rs`)

- **[MOVED+RESHAPE]** `ContextProvider` (ABC) → plain base
  (`_sessions.py:364`); `invoking(messages)->Context` → **`before_run(*, agent,
  session, context, state)`**, `invoked(...)` → **`after_run(...)`**, both now
  return `None` and mutate `SessionContext` in place; `thread_created` removed;
  `__init__(source_id)` for attribution.
- **[RENAME+RESHAPE]** `Context` → `SessionContext` (`_sessions.py:166`): from
  `{instructions, messages, tools}` to `{session_id, service_session_id,
  input_messages, context_messages: dict[source_id, msgs], instructions, tools,
  middleware, response, options, metadata}` with
  `get_messages(*, sources, exclude_sources, include_input, include_response)`.
- **[REMOVED]** `AggregateContextProvider`.

**Rust actions (`memory.rs`):**
- Rename `ContextProvider::invoking`→`before_run`, `invoked`→`after_run`
  (`memory.rs:37,47`); add a `source_id` ctor arg; change return from
  `Result<Context>` to in-place mutation of a `SessionContext`.
- Rename `struct Context`→`SessionContext` (`memory.rs:16`); expand fields.
- **Delete `struct AggregateContextProvider`** (`memory.rs:65-95`) + its impl;
  replace with agent-side iteration over `Vec<Arc<dyn ContextProvider>>`. Fix
  consumers: `agent.rs:15` import, `ChatAgent.context_provider`
  (`agent.rs:306`), builder (`agent.rs:1079`), `threads.rs:143`, and the
  `cp.invoking(...)`/`cp.invoked(...)` call sites (`agent.rs:422,432,510`).
  Update prelude (`lib.rs:58`). Note `agent-framework-redis`'s
  `RedisContextProvider` also implements this trait.

### §4 — Chat client (`_clients.py` → `client.rs`)

- **[RENAME]** `ChatClientProtocol`→`SupportsChatGetResponse` (`_clients.py:85`),
  now `Protocol[OptionsContraT]`.
- **[SIGNATURE]** `get_response` unified (Theme B); flat options → one `options`
  object; new per-call `compaction_strategy`/`tokenizer`; single abstract
  `_inner_get_response(*, stream)`.
- **[NEW]** `BaseChatClient` fields: **`STORES_BY_DEFAULT: ClassVar[bool]`**
  (drives auto-injection of `InMemoryHistoryProvider` unless the service stores
  server-side, `_clients.py:283`), `compaction_strategy`, `tokenizer`,
  `OTEL_PROVIDER_NAME`.
- **[NEW]** `@runtime_checkable` capability protocols: `SupportsCodeInterpreterTool`
  (`:668`), `SupportsWebSearchTool` (`:698`), `SupportsImageGenerationTool`
  (`:728`), `SupportsMCPTool` (`:758`), `SupportsFileSearchTool` (`:789`),
  `SupportsShellTool` (`:819`).
- **[NEW]** embeddings: `SupportsGetEmbeddings` (`:871`) + `BaseEmbeddingClient`
  (`:926`).
- **[REMOVED]** `create_agent` (see §1). No long-running/background/poll methods
  were added to the client; the only async-surface additions are `ResponseStream`
  and service-managed sessions via `ServiceSessionId`.

**Rust actions (`client.rs`):**
- Rename `trait ChatClient`→`SupportsChatGetResponse` (or keep the ergonomic
  name but align semantics to the Python export) and fold
  `get_streaming_response` into `get_response(stream)`: trait methods
  (`:35,42`), blanket impl (`:57,64`), `FunctionInvokingChatClient` (`:368,613`),
  `RetryingChatClient` (`:976,1001`), internal `inner_get_response` (`:122`).
- Replace flat-option args with an `options` object; add `compaction_strategy`/
  `tokenizer`; add `STORES_BY_DEFAULT` + the auto-inject rule.
- Add the six `Supports*Tool` capability traits and `SupportsGetEmbeddings`/
  `BaseEmbeddingClient` if embeddings are in scope. `RetryPolicy` structure
  carries over.

### §5 — Core types (`_types.py` → `types/{content,message,options,response}.rs`)

Keep the Rust `Content` enum-of-structs (`content.rs:320`) — it is the idiomatic
equivalent of the new single `Content` class and arguably better. Absorb:

- **[NEW]** twelve `ContentType` variants (`_types.py:345-369`):
  `code_interpreter_tool_call`/`_result`, `image_generation_tool_call`/`_result`,
  `mcp_server_tool_call`/`_result`, `search_tool_call`/`_result`,
  `shell_tool_call`/`_result`, `shell_command_output`, `oauth_consent_request`.
  Add matching enum variants + structs in `content.rs`. **Note:** the existing
  `Content::Unknown` fallback (`content.rs:338`) means old Rust deserializes new
  payloads inertly instead of crashing — good, but it **silently drops**
  shell/MCP/image content until the variants exist.
- **[SIGNATURE]** `UsageDetails` gains `cache_creation_input_token_count`,
  `cache_read_input_token_count`, `reasoning_output_token_count`
  (`_types.py:400`). Add them as first-class `Option<u64>` (they currently land
  untyped in `additional_counts`; wire-compatible but untyped). Keep the `Add` impl.
- **[RENAME]** `ChatMessage`→`Message` (`message.rs:63`, `mod.rs:16`).
- **[RENAME+SIGNATURE]** `AgentRunResponse`→`AgentResponse` (+`agent_id`,
  `continuation_token`), `AgentRunResponseUpdate`→`AgentResponseUpdate`
  (`response.rs:333,427`). Rust's `value`/`parse_json::<T>()` stands in for
  `.value`/generics.
- **[SIGNATURE]** `ChatResponse.model_id`→`model` + add `continuation_token`
  (`response.rs:47`).
- **[SIGNATURE]** `ChatOptions.model_id`→`model` (`options.rs:209`); `ToolMode`
  gains **`allowed_tools`** (`options.rs:132`).
- **[NEW]** `ContinuationToken` (`_types.py:2137`) — opaque provider token for
  background/resumable runs; add as a newtype on `ChatResponse`/`AgentResponse`
  and thread through run/streaming. Genuinely new capability.
- **[SIGNATURE]** `CitationAnnotation`→`Annotation` + `type` discriminator tags
  on annotations/regions (`content.rs:60-83`) — wire-format change.
- **[NEW]** embedding types (`EmbeddingGenerationOptions`, `Embedding`,
  `GeneratedEmbeddings`) — new `types/embedding.rs` if embeddings are in scope,
  else defer.
- **[KEPT]** `Role`/`FinishReason` already open string newtypes — no action.

### §6 — Tools (`_tools.py` + `tools/` + `_skills.py` → `tools.rs`)

- **[RENAME]** `AIFunction`→`FunctionTool`; `ai_function`→`tool` decorator, which
  gains `schema=`, `kind=`, `result_parser=` (`_tools.py:1150`). Rust `AiFunction`
  (`tools.rs:402`) already has `approval_mode`/`max_invocations`/
  `max_invocation_exceptions`; rename to `FunctionTool` and add `kind` (the
  `ToolKind` enum already exists, `tools.rs:87`), explicit `schema`, `result_parser`.
  `ApprovalMode` (`tools.rs:105`) matches upstream — [KEPT].
- **[REMOVED→replaced]** the concrete `Hosted*Tool` classes were removed;
  hosted capabilities are now the `Supports*Tool` client protocols (§4) + the
  new `Content` tool-call/result variants (§5). Rust already models hosted tools
  as factory functions returning `ToolDefinition` (`tools.rs:697-751`) — the
  right spirit. **Add `hosted_image_generation()` and shell-tool support.**
- **[MOVED]** tool approval → `_harness/_tool_approval.py`
  (`ToolApprovalMiddleware`, `ToolApprovalRule`, `ToolApprovalState`), requires
  an `AgentSession`. Rust has per-tool `ApprovalMode` but no standing-rule
  middleware — add once sessions land.
- **[NEW]** **Skills** (`_skills.py`, 4370 lines) — progressive-disclosure
  capability packages surfaced through three framework-generated `FunctionTool`s
  (`load_skill`/`read_skill_resource`/`run_skill_script`) and attached via a
  `SkillsProvider(ContextProvider)`. New `skills` module, **gated on §3**.
  Medium-large; MCP skills are `@experimental`.
- **[NEW]** `tools/` package (`agent_framework_tools`) — `LocalShellTool`,
  `DockerShellTool`, `ShellPolicy` (a UX pre-filter, *not* a security boundary),
  each producing a `FunctionTool` of `kind="shell"`. New separate crate; depends
  on the shell `Content` variants (§5).

### §7 — Middleware (`_middleware.py` → `middleware.rs`)

- **[RENAME]** `AgentRunContext`→`AgentContext` (`_middleware.py:93`,
  `middleware.rs:96`).
- **[SIGNATURE]** contexts dropped `SerializationMixin`; `ChatContext` gained
  `client`, `session`, and streaming hooks
  (`stream_transform_hooks`/`stream_result_hooks`/`stream_cleanup_hooks`), and
  `result` now admits a `ResponseStream`. Rust contexts (`middleware.rs:96-160`)
  already carry `metadata`/`result`/`terminate` — **add `session`, `client`, and
  the stream-hook fields**.
- **[NEW]** `MiddlewareTermination` exception for clean short-circuit — Rust's
  `ctx.terminate = true` (`middleware.rs:103/125/147`) is the idiomatic
  equivalent; **no action**.
- **[MOVED]** `use_*_middleware` decorators → `*MiddlewareLayer` mixins (Theme D)
  — Rust's `MiddlewarePipeline` is the equivalent; **no structural action**.
- **[NEW]** `AgentLoopMiddleware` (from the harness, §9).

### §8 — Observability (`observability.py` / `_telemetry.py` → `observability.rs`)

- **[MOVED]** `use_observability`/`use_agent_observability` →
  `ChatTelemetryLayer`/`AgentTelemetryLayer`/`EmbeddingTelemetryLayer`;
  `setup_observability` → `enable_instrumentation` + `configure_otel_providers`.
  Rust's `ObservableChatClient` wrapper substitutes for the layers; **add
  `enable_instrumentation`/`disable_instrumentation`/`enable_sensitive_telemetry`
  entry points and an agent-level telemetry wrapper.**
- **[NEW]** `OtelAttr` gained `CACHE_CREATION_INPUT_TOKENS`,
  `CACHE_READ_INPUT_TOKENS`, `REASONING_OUTPUT_TOKENS` (`observability.py:203`),
  `EMBEDDING_OPERATION="embeddings"` (`:292`), `PROMPT_NAME` (`:302`), and a
  `USAGE_DETAIL_TO_OTEL_ATTR` map. **Add the three cache/reasoning attrs (pair
  with §5 `UsageDetails`), the `embeddings` operation, and `prompt.name`.** The
  core `gen_ai.*` set + metric names already match — [KEPT].
- **[NEW]** MCP telemetry helpers `create_mcp_client_span`/`set_mcp_span_error`
  — add when the MCP client is ported.
- **[RENAME]** `ChatMessageListTimestampFilter`→`MessageListTimestampFilter`.
- Optional: `get_user_agent()` + a `foundry-hosting` UA prefix (low priority).

### §9 — New core modules

- **[NEW] `_compaction.py`** (1506 lines, **stable, not experimental**) —
  conversation-history compaction by annotation (never deletes; flags
  `_excluded`). `TokenizerProtocol` (`:51`), `CompactionStrategy` (`:60`), seven
  strategies (`Truncation`, `SlidingWindow`, `SelectiveToolCall`, `ToolResult`,
  LLM `Summarization`, `TokenBudgetComposed`, `ContextWindow`),
  `CompactionProvider(ContextProvider)` (`:1232`), wired into the client via
  `get_response(compaction_strategy=, tokenizer=)`. **A real functional gap.**
  New `compaction.rs` + `TokenizerProtocol` trait + `CompactionProvider`.
- **[NEW] `_settings.py`** (293) — replaces `_pydantic.py`;
  `pydantic-settings`/`AFBaseSettings` dropped for function-based
  `load_settings(...)` (override → `.env` → env → default) + a `repr`-masking
  `SecretString`. Rust: formalize env+dotenv precedence + a `Debug`-masking
  `SecretString` newtype. Low effort.
- **[NEW] `_feature_stage.py`** (403) — `@experimental(feature_id)` /
  `@release_candidate` decorators emitting one-time warnings; used 76× across
  core. Rust: map to `#[doc]` stability notes / a feature-id registry. Mainly a
  signal for which new modules are unstable (EVALS, FIDES, HARNESS, MCP_SKILLS).
- **[NEW] `_evaluation.py`** (2117, `@experimental`) — provider-agnostic
  agent/workflow eval (`Evaluator`, `LocalEvaluator`, `evaluator` decorator,
  `evaluate_agent`/`evaluate_workflow`). Low/optional; port later.
- **[NEW] `_harness/`** (11 files, `@experimental`) — a Claude-Code-style
  autonomous coding harness. `create_harness_agent(...)` composes a standard
  `Agent` + `AgentLoopMiddleware` + seven `ContextProvider`s (`TodoProvider`,
  `AgentModeProvider`, `FileAccessProvider`, `FileMemoryProvider`,
  `MemoryContextProvider`, `BackgroundAgentsProvider`) + `ToolApprovalMiddleware`.
  Large but optional; implementable as composition once §3/§7 land.
- **[NEW] `security.py`** (3532, `@experimental(FIDES)`) —
  information-flow-control / taint-tracking prompt-injection defense (labels,
  `ContentVariableStore`, two `FunctionMiddleware`, quarantine-LLM tool,
  `SecureAgentConfig`, `SecureMCPToolProxy`). High effort if adopted; experimental
  and self-contained.
- **[NEW] `_docstrings.py`** (110) — Python `__doc__` tooling; **no Rust action.**

### §10 — Workflow engine: builder / graph / edges / validation / viz

> **Convergence note.** The Rust workflow engine independently landed several
> designs upstream only reached in this window — do **not** re-churn these:
> `WorkflowBuilder` already requires a start executor and never had
> factory/string-ID registration (Python just removed both); `events.rs` is
> already a single discriminated-union `enum WorkflowEvent` (Python just
> collapsed 12 event classes into one); `WorkflowMessage` is already named that
> (Python just renamed `Message`→`WorkflowMessage`); `sub_workflow.rs` already
> implements only the `propagate_request=True` mode (now one of upstream's two
> sanctioned modes); `checkpoint.rs` already carries a `graph_signature` field;
> and `sequential.rs` already has no adapter executors. The actions below are
> the genuine gaps that remain.

**[NEW — correctness] Per-executor serialization within a superstep**
(`868744ae`, PR #6776, merged 2026-07-06 — the newest change in this whole
re-baseline). `Executor.execute()` now wraps its body in a lazily loop-bound
`self._execution_lock` so concurrent deliveries **to the same executor
instance** run one at a time, while different executors stay concurrent
(`_executor.py:211,232,271`). Rust's `runner.rs` pushes one `Invocation` per
message and drives them all through a single `futures::future::join_all`
(`runner.rs:660-668,695-703,712-721`), so two `execute()` calls can be in flight
on the same `Arc<dyn Executor>` at once — a real determinism/correctness gap for
any stateful executor that can receive 2+ messages in one superstep. **Rust
action:** group invocations by executor id and run each group sequentially,
while still `join_all`-ing across distinct ids.

**[REMOVED→REDESIGN] Event model unified** (`0f3f4dbc`, PR #3690). All 12 event
subclasses (`WorkflowStartedEvent`, `RequestInfoEvent`, `WorkflowOutputEvent`,
`ExecutorInvoked/Completed/Failed`, `AgentRunUpdateEvent`, …) collapsed into one
`WorkflowEvent(Generic[DataT])` with a `type: WorkflowEventType` string
discriminator. Rust's `events.rs:36-74` is **already** a single enum (a genuine
convergence), but its variant set now trails upstream: no `Intermediate` (new,
non-terminal progress output — §12 cross-cutting), no `ExecutorBypassed` (from `_functional.py`,
below), no `GroupChat`/`HandoffSent`/`MagenticOrchestrator` typed payloads, and
still-dedicated `AgentRunUpdate`/`AgentRun` variants where upstream now routes
agent output through generic `output`/`intermediate`. **Rust action:** add the
missing variants; `WorkflowEvent::Intermediate{..}` is the highest-value one
(Magentic currently proxies orchestrator messages through `AgentRunUpdate` for
lack of it).

**[NEW — experimental] Functional workflows (`_functional.py`, 1553 lines).** A
`@workflow`-decorator alternative to graph construction (`FunctionalWorkflow` /
`FunctionalWorkflowAgent`, deliberately **not** subclassing `Workflow`), with
optional `@step`/`RunContext` per-step caching, checkpointing, and HITL. Gated
`@experimental(FUNCTIONAL_WORKFLOWS)`. **Rust action:** none now — orthogonal
(no graph/`Executor`/`Edge` reuse) and experimental; track as a possible future
`#[workflow_step]`-style macro.

**`_workflow_builder.py`** (full API redesign):
- **[SIGNATURE]** `WorkflowBuilder.__init__` now requires keyword-only
  `start_executor`, and gained `checkpoint_storage`, `output_from`,
  `intermediate_output_from`, `output_executors` (deprecated alias)
  (`_workflow_builder.py:89`).
- **[REMOVED]** `register_executor()`, `register_agent()`, `add_agent()`
  (string-name / lazy-factory registration — the builder no longer defers
  executor construction), public `set_start_executor()` (now private, called
  from `__init__`), `set_max_iterations()` and `with_checkpointing()` fluent
  methods (both constructor-only now), and the five deferred-registration
  dataclasses.
- **[SIGNATURE]** every graph-building method (`add_edge`, `add_fan_out_edges`,
  `add_switch_case_edge_group`, `add_multi_selection_edge_group`,
  `add_fan_in_edges`, `add_chain`) narrowed params from
  `Executor | AgentProtocol | str` → `Executor | SupportsAgentRun` — **string
  executor-ID references are no longer accepted anywhere.**
- **[NEW]** workflow-output designation (`output_from`/`intermediate_output_from`
  + `_coerce_*`/`_resolve_designated_executor_ids`/`_validate_designation_lists`).
- **[BEHAVIOR]** `build()` emits a `DeprecationWarning` when neither
  `output_from` nor `intermediate_output_from` is set.

**`_workflow.py`:**
- **[REMOVED+MERGED]** `run_stream()`/`send_responses()`/
  `send_responses_streaming()` → one overloaded `run(message=None, *,
  stream=False, responses=None, checkpoint_id=None, checkpoint_storage=None,
  include_status_events=False, …)` (`_workflow.py:668`). **Single biggest
  workflow break.**
- **[SIGNATURE]** concurrency guard: boolean `_is_running` → `weakref.ref`-based
  `_active_run` (a dropped, un-iterated stream releases the run lock via GC).
- **[SIGNATURE]** `Workflow.__init__` reordered; `name` became required.
- **[NEW]** `OutputDesignation` dataclass; `get_output_executors()`/
  `get_intermediate_executors()`/`is_terminal_executor()`/…; `status` property
  (live `WorkflowRunState`); `WorkflowRunResult.get_intermediate_outputs()`.
- **[SIGNATURE]** `as_agent()` gained `description`, `context_providers`,
  `**kwargs`; `to_dict()` gained `output_executors`/`intermediate_executors`.
- **[VISIBILITY]** `graph_signature`/`graph_signature_hash` now public attrs.

**`_edge.py` / `_edge_runner.py`:**
- **[SIGNATURE]** `Edge.should_route()` sync → **async**
  (`EdgeCondition = Callable[[Any], bool | Awaitable[bool]]`); all
  `EdgeRunner.send_message()` impls now `await ….should_route(...)`.
- **[NEW]** `Edge.has_condition`; **[BEHAVIOR]** `condition_name` only
  auto-derived when `None` (old always overwrote).
- **[SIGNATURE]** `Case.target`/`Default.target`: `Executor | str` →
  `Executor | SupportsAgentRun`.

**`_validation.py`:**
- **[REMOVED]** `ValidationTypeEnum.INTERCEPTOR_CONFLICT` +
  `InterceptorConflictError`.
- **[NEW]** `ValidationTypeEnum.OUTPUT_VALIDATION` + `_output_validation()`
  (rejects output/intermediate overlap, requires listed IDs to exist and to
  declare `workflow_output_types`).
- **[SIGNATURE]** `validate_workflow_graph()` gained required
  `output_executors: list[str]` + optional `intermediate_executors`;
  `WorkflowValidationError` base `Exception` → `WorkflowException`.

**`_viz.py`:**
- **[SIGNATURE]** all exporters (`to_digraph`/`export`/`save_svg`/`save_png`/
  `save_pdf`/`to_mermaid`) gained `include_internal_executors: bool = False`,
  filtering `InternalEdgeGroup` edges by default. No formats removed.

**Rust actions (`workflow/`):** make `WorkflowBuilder::new` take
`start_executor`; remove `register_executor`/`register_agent`/`add_agent`/
`set_max_iterations`/`with_checkpointing` fluent methods (fold into ctor); drop
string-ID edge references (narrow to executor/agent); add the
`output_from`/`intermediate_output_from` designation feature end-to-end
(builder → validator → `Workflow` → viz); make `should_route` async; unify
`Workflow::run_stream`/`send_responses` into `run(stream)`; add the
`OutputDesignation`/`status`/intermediate-output surface; add
`include_internal_executors` to the viz exporters.

### §11 — Workflow checkpoint / state / runner (**breaking on-disk format**)

- **[SIGNATURE] `WorkflowCheckpoint`** rewritten (`_checkpoint.py:71`):
  `workflow_id` → required `workflow_name`; new required `graph_signature_hash`;
  new `previous_checkpoint_id` (chaining); `shared_state`→`state`; `messages`
  now holds live `WorkflowMessage` objects (not pre-serialized dicts);
  `pending_request_info_events` now holds `WorkflowEvent`. `version` stays
  `"1.0"` **despite the breaking change** — cannot be used to detect old files.
- **[RENAME] `CheckpointStorage`**: `save_checkpoint`→`save`,
  `load_checkpoint`→`load`, `delete_checkpoint`→`delete`; new
  `get_latest(*, workflow_name)`; filter param `workflow_id` (optional) →
  `workflow_name` (**required, keyword-only**).
- **[SIGNATURE]** `load()` no longer returns `Optional` — raises
  `WorkflowCheckpointException` on miss.
- **[NEW]** `FileCheckpointStorage(storage_path, *, allowed_checkpoint_types)`
  with `_validate_file_path()` (blocks path traversal — a real security fix);
  `InMemoryCheckpointStorage.save()` now `deepcopy`s before storing.
- `_checkpoint_encoding.py` **full rewrite**: custom markers +
  lossy `str(v)` → pickle+base64 for non-JSON-native values, gated by a
  `_RestrictedUnpickler` allowlist (documented as defense-in-depth, **not** an
  RCE boundary). **OLD checkpoint files are unreadable by NEW code, no migration
  path.**
- **[REMOVED]** `_checkpoint_summary.py` (`WorkflowCheckpointSummary`,
  `get_checkpoint_summary`) — deleted, not merged.
- **[RENAME] `SharedState`→`State`** (`_shared_state.py` deleted, `_state.py`
  new) and **now internal-only** (not exported). Methods un-`async`'d;
  `hold()`/`*_within_hold` transaction API removed; `get(key)` (raised
  `KeyError`) → `get(key, default=None)`. **[NEW] concurrency model**: no
  `asyncio.Lock`; a pending-buffer/commit model where `Runner` calls
  `state.commit()` **once per superstep** (barrier at the Pregel boundary
  instead of per-op locking). **The single most important state-model change to
  replicate.**
- **[REMOVED]** `_conversation_state.py` (`encode/decode_chat_messages`) —
  deleted.
- **[RENAME]** `Runner`→`RunnerImpl` (`Runner` kept as a deprecation shim);
  **[REMOVED]** reentrancy guard; **[NEW]** `state.commit()` per superstep;
  `restore_from_checkpoint(...)` now `-> None` (raises on failure) and strictly
  compares `graph_signature_hash`.
- **[BEHAVIOR]** `Workflow.run()` **no longer wipes state per call** — iteration
  count is not reset except via `reset_iteration_count()` or a checkpoint
  restore. **A Rust port assuming stateless one-shot `run()` must account for
  this.**
- **[RENAME]** `_runner_context.py`: `Message`→`WorkflowMessage`;
  `WorkflowMessage.original_request`→`original_request_info_event`
  (`WorkflowEvent`); `create_checkpoint(...)` gained `workflow_name`,
  `graph_signature_hash`, `state`, `previous_checkpoint_id`;
  `add_request_info_event` now takes `WorkflowEvent` (runtime-checks
  `type=="request_info"`); new yield-output classifier
  (`set_yield_output_classifier`/`classify_yielded_output`).
- `_workflow_context.py`: `shared_state`→`state`; async `get_shared_state`/
  `set_shared_state` (raised `KeyError`) → sync `get_state(key, default)`/
  `set_state`; new `request_id` param+property; `yield_output()` now classified
  (`output`/`intermediate`/hidden — `None` emits **no event**);
  `set_executor_state`/`get_executor_state` deleted.
- `_workflow_executor.py` (sub-workflow HITL): **[NEW]**
  `propagate_request: bool=False` — `True` forwards a sub-workflow's request as
  a real `request_info` event on the parent (correlated by preserving
  `request_id`); handler renames + a shared `_handle_response`.

**Rust actions:**
- `workflow/checkpoint.rs`: adopt the new `WorkflowCheckpoint` fields +
  `save`/`load`/`delete`/`get_latest` names + raise-on-miss + path-traversal
  guard + deepcopy-on-save; adopt full-fidelity value encoding with a type
  allowlist (Rust would use tagged serialization, not pickle); add
  `previous_checkpoint_id` chaining (additive, low priority). **Convergence:**
  Rust already has `graph_signature`, so the hash validation is close; and Rust
  keeps `WorkflowCheckpointSummary` (`checkpoint.rs:116`) which upstream dropped
  — fine to keep, but fix the stale doc citation at `checkpoint.rs:3` that
  claims it mirrors the now-deleted `_checkpoint_summary.py`.
- `workflow/shared_state.rs`: the **highest-value state change.** Today it uses
  a `tokio::sync::RwLock<HashMap>` with **immediate** `get`/`set` (`:32-39`), so
  a concurrent sibling in the same superstep sees another's writes
  non-deterministically — the exact race Python (and .NET) just closed. Add a
  pending/committed split and have `WorkflowRun::run_loop` call `commit()` once
  per superstep after `join_all` resolves and before `maybe_checkpoint`
  (`runner.rs:800`).
- Make `Workflow::run()` **not** reset state/iteration per call (`runner.rs:373`
  always builds a fresh `WorkflowRun`), so multi-turn `WorkflowAgent`
  conversations retain workflow-internal state — today they cannot.
- `workflow/context.rs:111`: add a public `request_info` `request_id` param
  (only the crate-private `record_request_with_id` supports pinning today).
- Add the yield-output classifier (`output`/`intermediate`/hidden) and the
  sub-workflow behavior; delete the conversation-state analog.
- `orchestration/workflow_agent.rs:279`: add `checkpoint_id`/`checkpoint_storage`
  options to `WorkflowAgent::run`/`run_stream` — upstream added agent-level
  checkpoint restore (`_agent.py:141`); Rust's `WorkflowAgent` cannot resume from
  a checkpoint at all today.

### §12 — Orchestrations (extracted to a separate package; not a file move)

All five patterns moved from `core/agent_framework/_workflows/` to a standalone,
separately-versioned, **optionally-installed** package `agent_framework_orchestrations`
(`0daa7700`, PR #3685; core depends on it only as an extra and keeps a lazy
re-export shim at the new public path `agent_framework.orchestrations`), and
**every one changed more than its location.**

> **Structural recommendation.** This argues for promoting the Rust
> `crates/agent-framework-core/src/workflow/orchestration/*` into its own crate
> (e.g. `agent-framework-orchestrations` depending on `agent-framework-core`):
> it mirrors upstream's new package boundary, and orchestration is the odd one
> out — every other domain in this workspace is already crate-per-package. The
> split looks mechanically clean (its `use crate::workflow::{…}` imports all
> resolve to already-`pub`-exported symbols), and would force a cleaner boundary
> than Python's own extraction, which still reaches into leading-underscore
> submodules of core. The port's `README.md:246` mapping table still cites the
> stale `_workflows._{sequential,concurrent,group_chat,handoff,magentic}` paths
> and should be updated to `agent_framework_orchestrations`.

Cross-cutting:

- **[SIGNATURE]** all builders narrowed participants to
  `Sequence[SupportsAgentRun | Executor]` (or `Agent`-only for GroupChat/Handoff);
  `AgentExecutorResponse.agent_run_response`→`.agent_response`.
- **[NEW — most pervasive addition] `output_from`/`intermediate_output_from`**
  participant-output designation (repurposed `_participant_utils.py` →
  `_participant_output_config.py`) lets callers pick which participants' own
  `yield_output` surfaces as workflow `output` vs. `intermediate` events —
  replacing hardcoded "only the last / only the aggregator yields" wiring and
  the removed per-`AgentExecutor` `output_response` flag. **No Rust orchestration
  builder has any equivalent** (verified across all five files). Implementing it
  requires, in order: (1) `WorkflowEvent::Intermediate{..}` (§10), (2) an
  `OutputDesignation` set on `WorkflowBuilder`/`Workflow`, (3) an
  `OutputValidation` check (`validation.rs`), (4) `.output_from(...)`/
  `.intermediate_output_from(...)` on each builder. Note Rust's
  `AgentExecutor::with_output(bool)` flag (`orchestration/mod.rs:171`) mirrors
  the *removed* Python design and should be superseded by this.

- **[SIGNATURE]** all builders narrowed participants to
  `Sequence[SupportsAgentRun | Executor]` (or `Agent`-only for GroupChat/Handoff);
  `AgentExecutorResponse.agent_run_response`→`.agent_response`.
- **[NEW]** `output_from`/`intermediate_output_from` on Sequential/Concurrent/
  GroupChat/Magentic (a `_participant_output_config.py` selecting which
  participants' `yield_output` surfaces as workflow output/intermediate events).
- **[REMOVED→replaced] generic HITL engine**: `RequestInfoInterceptor`
  (one class, pre-agent, bare-`str` response) → **`AgentApprovalExecutor`**
  wrapping a two-node sub-workflow that pauses **after** the agent responds;
  response is a structured `AgentRequestInfoResponse` (empty = approve, non-empty
  = re-invoke) — enabling **iterate-until-approved loops that did not exist
  before.** Used by Sequential/Concurrent/GroupChat; Handoff and Magentic keep
  bespoke request/response types.
- **[BEHAVIOR]** terminal output broadly shifted from raw `list[ChatMessage]`
  conversation dumps to `AgentResponse`/`AgentResponseUpdate`.
- **[REMOVED]** `_participant_utils.py` (`wrap_participant`, `sanitize_identifier`,
  `build_alias_map`, …) wholly deleted.

**Sequential** — `SequentialBuilder` [KEPT] name; ctor-kwargs shift
(`participants=`, `checkpoint_storage=`, `chain_only_agent_responses=`,
`output_from=`); `register_participants()`/`with_checkpointing()` removed;
`.with_request_info()` moved from pre-agent to post-agent approval; terminal
output `list[ChatMessage]`→`AgentResponse`; **[NEW]** `chain_only_agent_responses`
→ `AgentExecutor(context_mode="full"|"last_agent"|"custom")`. Graph topology
unchanged.

**Concurrent** — `ConcurrentBuilder` [KEPT]; same ctor-kwargs shift;
`.with_request_info()` gained a per-agent `agents=` filter and now wraps **each**
participant individually (N per-agent pauses replace 1 combined pause); default
aggregator output `list[ChatMessage]` (with prompt) → `AgentResponse` (no prompt,
one message per participant). (The old `.with_custom_aggregator` reference was a
stale OLD docstring — the method was always `with_aggregator`.)

**GroupChat** (the significant redesign) — `GroupChatBuilder` [KEPT] name, but
the one dual-mode `GroupChatOrchestratorExecutor` split into **`GroupChatOrchestrator`**
(callable `selection_func`) and **`AgentBasedGroupChatOrchestrator`** (structured
`AgentOrchestrationOutput`). "Manager"→"Orchestrator" throughout;
`set_manager`/`set_select_speakers_func` fluent methods → ctor kwargs
`orchestrator_agent=`/`orchestrator=`/`selection_func=`. Field rename
`selected_participant`→`next_speaker`, `finish`→`terminate`; **`instruction`
field [REMOVED]** (a non-empty `instruction` used to be injected before the next
speaker — that capability is gone). Many symbols deleted (`GroupChatDirective`,
`ManagerSelectionRequest/Response`, `GroupChatStateSnapshot`,
`DEFAULT_MANAGER_*` prompts). `ParticipantRegistry` moved into
`_base_group_chat_orchestrator.py` and lost its entry/exit-ID "pipeline"
abstraction. **[NEW]** typed events `GroupChatRequestSentEvent`/
`GroupChatResponseReceivedEvent`. **[BEHAVIOR]** terminal output changed from
the full transcript to just the final completion message; arbitrary-`Executor`-
as-manager no longer allowed. Topology (hub-and-spoke) unchanged.

**Handoff** — `HandoffBuilder` [KEPT] name; **alone among the five it stayed
fluent** (checkpointing/termination still settable via `.with_*()`). Methods
11→7: `set_coordinator`→`with_start_agent` [RENAME] (not equivalent —
coordinator was a mandated routing hub); `add_handoff` lost `tool_name`
(names now `get_handoff_tool_name()`-derived) and renamed `tool_description`→
`description`; `auto_register_handoff_tools`/`request_prompt`/
`enable_return_to_previous`/`with_request_info` **[REMOVED]**;
`with_interaction_mode(enum)`→`with_autonomous_mode(agents=, prompts=,
turn_limits=)` (global toggle → per-agent dict). `HandoffUserInputRequest`
(4 fields, untyped response) → `HandoffAgentUserRequest` (**1 field**
`agent_response`, strongly-typed `list[Message]` response) — **not the same
concept renamed.** **[NEW]** `HandoffConfiguration`, `HandoffSentEvent`,
`HandoffAgentExecutor` (per-participant). **[BEHAVIOR — topology change]**
hub-and-spoke → **fully-connected mesh**; default routing (when `add_handoff`
is never called) flipped hub-only → full mesh; `clean_conversation_for_handoff`
now keeps **only `text`** content (silently drops images); the **default
10-message termination safety net was removed** (workflows now run indefinitely
unless the caller sets a termination condition); `participants()` narrowed to
`Agent`-only (rejects bare custom `Executor`s — a real capability loss).

**Magentic** — `MagenticBuilder` [KEPT] name; no-arg ctor → large kwargs surface
(`participants=`, `manager=`/`manager_factory=`/`manager_agent=`/
`manager_agent_factory=` — the two `*_factory` modes are **[NEW]**, all prompt/
limit overrides, `enable_plan_review=`, `checkpoint_storage=`, `output_from=`).
`.participants(**kwargs)` and `.with_standard_manager(...)` removed as public API;
`.with_human_input_on_stall()` **[REMOVED]** (only a stale docstring remains);
`.start_with_string/_message/_with()` + the `MagenticWorkflow` wrapper removed
(use generic `Workflow.run()`). `MagenticManagerBase`/`StandardMagenticManager`
near-[KEPT] (prompts textually identical) but the manager gained session state
(`AgentSession`, checkpointed via a new `"agent_session"` field).
**HITL redesign**: plan-review + stall **merge** into
`MagenticPlanReviewRequest`/`Response` (binary approve/revise); tool approval
**[MOVED]** out to the shared generic `AgentExecutor` machinery. **Capability
loss:** old `enable_stall_intervention` let a human pick CONTINUE/GUIDANCE at a
stall; new always auto-resets+replans and only exposes approve/revise. `MagenticOrchestratorExecutor`→`MagenticOrchestrator` [RENAME]; `MagenticAgentExecutor`
rebased on the shared `AgentExecutor` (~390→~38 lines); graph now built directly
via `WorkflowBuilder`/`add_edge` (was `GroupChatBuilder` callback injection) —
Magentic is **decoupled from GroupChat**; **[NEW]** typed
`MagenticOrchestratorEvent` (`PLAN_CREATED`/`REPLANNED`/
`PROGRESS_LEDGER_UPDATED`).

**Rust actions (`workflow/orchestration/`):** a large, coordinated rework, but
weigh it against real convergences — several patterns already match new upstream
by independent design:

- **Sequential** — `sequential.rs:15-67` already has no adapter executors
  (`AgentExecutor::with_output(i == last)` does exactly what Python simplified
  to). Only the `output_from` designation (§12 cross-cutting) is genuinely new.
- **Concurrent** — functional shape matches; the gaps (`with_aggregator`,
  per-agent `with_request_info` filter) were already tracked pre-baseline, now
  with precise target signatures.
- **GroupChat** — Rust's private `GroupChatOrchestrator` **already** covers both
  new-Python modes (function-selection and LLM-agent-selection) via one struct
  dispatching to a `GroupChatManager` trait object, so **no subclass restructure
  is needed** — Rust achieves by composition what Python now does by
  subclassing. Real gaps: `with_checkpointing`/`with_request_info` on the builder;
  drop the `instruction` field; require native structured output for the
  agent-based manager (robustness).
- **Handoff** — the larger rebuild: mesh topology + per-participant
  `HandoffAgentExecutor`, per-target `HandoffConfiguration` (Rust uses bare target
  strings today), agent-object `with_start_agent`, and a **conscious decision**
  on the removed 10-message safety net. The response-inspection-vs-middleware
  divergence Rust already documents (`handoff.rs:14-19`) is unchanged upstream —
  still accurate.
- **Magentic** — Rust's `magentic.rs` implements the **old two-track HITL**
  (separate `MagenticPlanReviewRequest` + `MagenticStallInterventionRequest`,
  with an extra `edited_plan` capability **neither Python version has**). This is
  now a three-way divergence, and Rust's split model is arguably richer — **no
  forced rename is warranted**, but the doc comments citing `_magentic.py` line
  numbers now point at a moved+rewritten file; update them to
  `agent_framework_orchestrations/_magentic.py` and note upstream diverged into a
  merged shape rather than implying Rust mirrors it.

Genuinely new work regardless: the shared post-agent `AgentApprovalExecutor`
HITL engine (iterate-until-approved) for Sequential/Concurrent/GroupChat; wiring
`output_from`/`intermediate_output_from` through all four applicable builders;
and the terminal-output shape change (`AgentResponse` instead of transcript
dumps), which touches the orchestration tests.

### §13 — Providers

**`agent-framework-openai`** — [MOVED] already standalone (matches upstream).
- **[RENAME]** the `OpenAIChatClient` name **flipped**: it is now the **Responses**
  API client (`openai/_chat_client.py:3219`); the old Chat-Completions behavior
  is now **`OpenAIChatCompletionClient`**; `OpenAIResponsesClient` no longer
  exists.
- **[REMOVED]** `OpenAIAssistantsClient` (Assistants / `beta.threads`) — **gone.**
- **[NEW]** `OpenAIEmbeddingClient`; content-filter error types
  (`OpenAIContentFilterException`, `ContentFilterResult*`); Responses-specific
  typed options (`include`, `reasoning`, `verbosity`, `prompt_cache_key`,
  `safety_identifier`, …) + `OpenAIContinuationToken`; a `Raw*`/layered split;
  `OpenAISettings`/`AzureOpenAISettings` are now `TypedDict`s.
- **Rust actions:** rename `OpenAIClient`→`OpenAIChatCompletionClient`
  (`lib.rs:236`); promote `OpenAIResponsesClient` (`responses.rs`) to the
  canonical `OpenAIChatClient`; **remove `OpenAIAssistantsClient`
  (`assistants.rs`)** — upstream dropped the API; add an embedding client, the
  content-filter error type, and the new typed options.

**`agent-framework-azure`** — [BEHAVIOR/MOVED] the standalone `AzureOpenAI*Client`
classes were **deleted**; Azure OpenAI is now a **routing mode** of the unified
OpenAI clients (`OpenAIChatClient(azure_endpoint=, credential=, api_version=)`).
`AzureOpenAISettings` survives only as a `TypedDict`. **Rust action:** rework in
place (no rename) — realign as "Azure routing of the openai crate"; carry the
Chat/Responses name swap; Entra credential logic stays valid. Low churn.

**`agent-framework-azure-ai` → rename to `agent-framework-foundry`** (the
largest provider item). Upstream **deleted** `azure-ai` and replaced it with
`foundry`: `FoundryChatClient` (Responses API via `AIProjectClient`),
**`FoundryAgent`** ("connects to an existing PromptAgent or HostedAgent"), and
`to_prompt_agent()` → `PromptAgentDefinition` — exactly the Prompt-Agent client
the old gap analysis flagged as missing. Env prefix `AZURE_AI_`→`FOUNDRY_`,
`model_deployment_name`→`model`, `azure-ai-agents` SDK dropped; also new
`FoundryEmbeddingClient`, `FoundryMemoryProvider`, `FoundryEvals`. The current
Rust crate implements `AzureAIAgentClient` over the now-**removed** Agents/threads/
runs data-plane (`convert.rs`/`sse.rs`), i.e. it mirrors the deleted package.
**Rust action:** rename the crate and rewrite onto the Responses API +
`FoundryAgent`/`to_prompt_agent`; replace the threads/runs plumbing.

**`agent-framework-anthropic`** — [SIGNATURE/EXPANDED], not a rename. Same
package, now a superset: a `Raw*`/public split plus **multi-cloud transports in
the same package** — `AnthropicBedrockClient`, `AnthropicVertexClient`
(adds `project_id`), `AnthropicFoundryClient` — all delegated to the single
`anthropic` SDK. `model_id`→`model`; new `AnthropicChatOptions`. **Rust action:**
rework in place — add Bedrock (AWS SigV4), Vertex (Google ADC), and Foundry
client variants **in the same crate** (new auth deps); apply `model_id`→`model`.
Do **not** split the cloud transports into separate crates.

**New provider crates (no Rust equivalent today):**

| New crate | Wraps | Kind | Notes |
|---|---|---|---|
| `agent-framework-claude` | Claude Agent SDK / Claude Code CLI | `BaseAgent`, subprocess | Distinct from `anthropic` (agent, not chat client) |
| `agent-framework-bedrock` | AWS Bedrock Converse (`boto3`) | chat + embedding | namespace `amazon` |
| `agent-framework-gemini` | Google Gemini / Vertex (`google-genai`) | chat | `ThinkingConfig` |
| `agent-framework-mistral` | Mistral AI | **embeddings-only** | no chat client upstream |
| `agent-framework-ollama` | Ollama local | chat + embedding | |
| `agent-framework-github-copilot` | GitHub Copilot SDK | `BaseAgent` | like `claude`, not a chat client |
| `agent-framework-foundry-local` | on-device Foundry Local runner | chat (OpenAI-compatible) | |

**`agent-framework-copilotstudio`** — **docs-only.** Public surface is
unchanged; the D2E wire protocol the Rust port implements is unaffected. The
`run(stream=)` and `AgentSession`/`Message`/`AgentResponse` renames are inherited
from core. Two stale in-code notes must be fixed: the "Python starts a new
conversation unconditionally" divergence comment (`lib.rs:114-122`,
`agent.rs:186-190`) is now **false** — upstream matched the port's
reuse-then-create-on-first-use behavior — and the `settings.rs:23-24` comment
describing `AFBaseSettings` is stale (upstream is now `TypedDict`/`load_settings`).

### §14 — Hosting & ecosystem

**Hosting split (`hosting` + `hosting-responses`, both [NEW]).** These are *not*
a new server — they are small framework-agnostic libraries: `hosting` ships
`AgentState`/`WorkflowState`/`SessionStore` (target resolution + a plain session
store) and explicitly defers routing/auth/background-runs to the app's web
framework; `hosting-responses` is pure OpenAI-Responses-shape request/response
**conversion** (`responses_to_run`/`responses_from_run`/…). Protocol hosting now
lives **by protocol**: `a2a` bundles client **and** server (`A2AExecutor`),
`ag-ui` bundles client + server + its own snapshot store. Rust today has one
monolithic `agent-framework-hosting` crate serving all protocols, fully
stateless, with **no auth and no session store**, and two self-documented TODOs
already pointing the same direction (dedup `a2a.rs`'s duplicated wire types with
the client-only `agent-framework-a2a` crate; extract the Responses types
embedded in `devui/models.rs`). **Rust actions:** extract a reusable
Responses-conversion module (mirroring `hosting-responses`); move A2A server
hosting into `agent-framework-a2a` (or a shared types crate); a Rust `hosting`
crate matching `AgentState`/`SessionStore` is **blocked on core `AgentSession`**.

**`devui`.** Route set is byte-for-byte identical old→new (the conversations /
deployments API a naive audit would flag as "missing from Rust" already existed
at the Dec-2025 baseline — a pre-existing gap, not new drift). What changed:
**[NEW]** a `host_header` anti-DNS-rebinding middleware; **[SIGNATURE]**
`serve()` flipped `auth_enabled` default **`False`→`True`** (Bearer auth on by
default) and renamed `tracing_enabled`→`instrumentation_enabled`. Rust's `devui`
implements 4 of ~21 routes and has **zero** auth / Host-header checks. **Rust
actions:** port the auth-by-default + Host-header middleware (small,
security-relevant); implement the remaining ~17 routes against the new
`Message`/`AgentResponse`/`AgentSession` shapes (larger, pre-existing).

**`durabletask` [NEW package].** Durable agent + workflow hosting on Microsoft's
Durable Task framework (gRPC sidecar): `DurableAIAgentClient`/`Worker`/
`AgentEntity` and a `Workflow`→orchestrator/activity mapping. The bespoke
durable-state code `azurefunctions` used to carry inline was generalized into
this package. **Rust action: new crate** — substantial (gRPC sidecar client,
replay-safe orchestration model, entity model). Second-largest ecosystem gap
after declarative workflows.

**`azure-cosmos` [NEW package].** `CosmosHistoryProvider` (subclasses core
`HistoryProvider`) + `CosmosCheckpointStorage` (implements the workflow
`CheckpointStorage` protocol). Rust's `agent-framework-cosmos` today has only a
`chat_message_store.rs` on the **old** `ChatMessageStore` trait. **Rust actions:**
add a `CosmosCheckpointStorage` now (unblocked — the Rust `CheckpointStorage`
trait already exists at `workflow/checkpoint.rs:175`); the history-provider half
is blocked on core `HistoryProvider`.

**Existing ecosystem crates needing rework:**
- **`agent-framework-declarative`** — **largest ecosystem gap.** Upstream added a
  full declarative-**workflow execution engine** (`_workflows/`: control-flow,
  HTTP-request, MCP-tool, function-tool, external-input executors + PowerFx +
  a `DeclarativeWorkflowBuilder` compiling YAML into a real `Workflow`). Rust is
  **schema-only** (parses `WorkflowSpec`/`NodeSpec`/… but has no executor).
  New-crate-sized effort.
- **`agent-framework-redis`** — [RENAME+SIGNATURE]
  `RedisChatMessageStore`/`RedisProvider` → `RedisHistoryProvider`/
  `RedisContextProvider` (hooks `before_run`/`after_run`). Rust's names
  coincidentally match but mirror the **old** shapes; blocked on core.
- **`agent-framework-mem0`** — [RENAME] `Mem0Provider`→`Mem0ContextProvider`,
  same hook rename; blocked on core.
- **`agent-framework-purview`** — [SIGNATURE, minor] class names stable; add the
  callable token-provider credential option + session-id plumbing. Unblocked,
  lowest-effort.
- **`agent-framework-azure-ai-search`** — [MOVED file] hook `invoking`→
  `before_run`; blocked on core. The self-acknowledged "agentic/KB out of scope"
  gap is pre-existing, not new drift.

**New ecosystem packages with no Rust counterpart:** `azure-contentunderstanding`
(a `ContextProvider` doing OCR/transcription via Azure Content Understanding);
`monty` and `hyperlight` (two alternative CodeAct sandboxes — likely only one is
worth porting); `foundry_hosting` (hosts an AF agent behind the proprietary
Foundry Agent Server — skip unless a Rust `azure-ai-agentserver` SDK appears);
the `tools` package's `LocalShellTool` (small, unblocked — see §6).

---

## Part III — Consolidated action plan

### Breaking renames (Rust symbol → new name)

| Current Rust | New | Where |
|---|---|---|
| `trait Agent` | `SupportsAgentRun` | `agent.rs:217` + all `dyn Agent` sites |
| `struct ChatAgent` | `Agent` | `agent.rs:300` |
| `ChatAgentBuilder` | `AgentBuilder` | `agent.rs:989` |
| `AgentThread` | `AgentSession` | `threads.rs:111` |
| `ChatMessageStore` / `InMemoryChatMessageStore` | `HistoryProvider` / `InMemoryHistoryProvider` | `threads.rs:26,61` |
| `struct Context` | `SessionContext` | `memory.rs:16` |
| `ContextProvider::invoking` / `invoked` | `before_run` / `after_run` | `memory.rs:37,47` |
| `AggregateContextProvider` | *(deleted)* | `memory.rs:65` |
| `trait ChatClient` | `SupportsChatGetResponse` | `client.rs` |
| `ChatMessage` | `Message` | `message.rs:63` |
| `AgentRunResponse` / `...Update` | `AgentResponse` / `AgentResponseUpdate` | `response.rs:333,427` |
| `AiFunction` | `FunctionTool` | `tools.rs:402` |
| `AgentRunContext` | `AgentContext` | `middleware.rs:96` |
| `CitationAnnotation` | `Annotation` | `content.rs:60` |
| `OpenAIClient` | `OpenAIChatCompletionClient` | `openai/lib.rs:236` |
| `OpenAIResponsesClient` | `OpenAIChatClient` | `openai/responses.rs` |
| `SharedState` (workflow) | `State` (internal) | `workflow/` |
| crate `agent-framework-azure-ai` | `agent-framework-foundry` | Cargo + rewrite |

Field renames threading through many signatures: `ChatResponse.model_id`→`model`,
`ChatOptions.model_id`→`model`, `service_thread_id`→`service_session_id`,
`workflow_id`→`workflow_name` (checkpoints), Anthropic/Azure `model_id`→`model`.

### Removals (delete from the port; upstream no longer has them)

- **`OpenAIAssistantsClient`** (`openai/assistants.rs`) — API removed upstream.
  (The Wave-3 build + four PR-review rounds spent on it target a dead surface.)
- `AzureOpenAIAssistantsClient` convenience wrapper, if any.
- `AggregateContextProvider` (`memory.rs`).
- Workflow: `register_executor`/`register_agent`/`add_agent`, fluent
  `set_max_iterations`/`with_checkpointing`, checkpoint-summary, conversation-state.
- Orchestrations: Handoff `auto_register_handoff_tools`/`request_prompt`/
  `enable_return_to_previous`; GroupChat `instruction` field + `DEFAULT_MANAGER_*`;
  Magentic `with_human_input_on_stall`/`with_standard_manager`/`start_with_*`.

### Correctness fixes (independent of the rename cascade — do these anytime)

These are behavior/determinism gaps that upstream closed and the port can fix
without waiting on the big renames:

1. **Per-executor serialization within a superstep** (§10) — group
   same-executor invocations and run them sequentially in
   `runner.rs:660-721`; today `join_all` lets two `execute()` calls race on one
   `Arc<dyn Executor>`.
2. **`SharedState` pending-buffer / commit-per-superstep** (§11) — replace
   immediate `RwLock` writes (`shared_state.rs:32-39`) with a staged
   pending/committed split committed once per superstep (`runner.rs:800`); this
   is the determinism fix both Python and .NET already made.
3. **`Workflow::run()` retaining state across calls** (§11) — stop building a
   fresh `WorkflowRun` per call so multi-turn `WorkflowAgent` works.

### Ripple order (what unblocks what)

1. **Core type renames** (`ChatMessage`→`Message`, `AgentRunResponse`→
   `AgentResponse`, `model_id`→`model`) — touch every signature; do first.
2. **`trait Agent`→`SupportsAgentRun` + `ChatAgent`→`Agent`** — ~40 files;
   unblocks the `Agent` name.
3. **Unify `run_stream`→`run(stream)` / `get_streaming_response`→
   `get_response(stream)`** — every client/agent/workflow/orchestration call site.
4. **Sessions + memory merge** (`AgentSession`, `HistoryProvider`,
   `SessionContext`, `before_run`/`after_run`, delete `AggregateContextProvider`)
   — the keystone that unblocks **compaction, skills, security, harness,** the
   hosting `SessionStore`, and the redis/mem0/cosmos/azure-ai-search provider
   reworks.
5. **Workflow `State` (commit-per-superstep) + checkpoint format** — unblocks the
   cosmos checkpoint store and correct resume semantics.
6. **Providers + orchestrations** — mostly mechanical once 1–4 land.

### New crates to add / split out

Providers: `agent-framework-{claude, bedrock, gemini, mistral, ollama,
github-copilot, foundry-local}`. Ecosystem: `agent-framework-durabletask`,
a `agent-framework-tools` (shell) crate, and (rename) `agent-framework-foundry`.
Possibly `agent-framework-foundry-hosting` and one CodeAct sandbox crate.
**Structural:** promote `workflow/orchestration/*` out of `agent-framework-core`
into its own `agent-framework-orchestrations` crate, mirroring upstream's new
package boundary (§12).

### New capabilities worth adopting (ranked)

1. **Sessions / `SessionContext`** (§2/§3) — the keystone; cheap relative to
   leverage.
2. **History compaction** (§9) — stable, broadly useful, wired into the core
   client loop. A real functional gap.
3. **New `Content` tool-call variants + hosted image-gen/shell tools** (§5/§6) —
   parity with modern providers; the `Content::Unknown` fallback silently drops
   these today.
4. **`ContinuationToken` / background-resumable runs** (§5).
5. **Foundry Prompt-Agent client** (§13) — the gap the old analysis flagged,
   now shipping upstream.
6. **Shell-tools crate** (§6) and **cache/reasoning usage telemetry** (§5/§8).
7. **Skills** (§6) — valuable but large, depends on sessions.
8. **Harness / security / evaluation** (§9) — large, all `@experimental`; adopt
   opportunistically after the foundations.

---

## Part IV — What this invalidates in the existing docs

`GAP_ANALYSIS.md` and `PARITY.md` are both written against `638fbb5f`. Specific
claims that this re-baseline overturns, to be corrected in the companion pass:

- The `ChatAgent` naming is presented as matching upstream — it no longer does
  (`Agent` is the upstream name).
- The OpenAI/Azure **Assistants** clients are tracked as parity wins — the API
  was removed upstream; they are now dead surface.
- The "missing Foundry Prompt-Agent client" gap is **filled** upstream (the
  `foundry` package) and should move from "missing in Rust" to "rename + rewrite
  the existing `azure-ai` crate".
- Provider/orchestration/hosting **package layout** claims (all-in-core vs.
  extracted) are stale.
- Whole new surfaces no matrix row tracks: seven new provider packages,
  sessions/skills/compaction/harness/security, `durabletask`, `azure-cosmos`,
  the declarative-workflow engine, `hosting`/`hosting-responses`, DevUI
  auth-by-default.
