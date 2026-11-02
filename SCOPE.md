# Scope

What this port deliberately does not carry from the official
[Microsoft Agent Framework](https://github.com/microsoft/agent-framework)
(Python and .NET) and [Agent Framework for Go](https://github.com/microsoft/agent-framework-go),
and why. The weekly drift pass cites this file instead of re-deriving the
same reasons each window; see [`docs/drift/`](docs/drift/README.md).

There are three lists. Only the first is a decision not to port. The second
is upstream behaviour expressed differently, and the third is work still
open. Anything not listed is in scope.

Reopening an item means changing this file in the same PR that ports it.

## Out of scope

| Upstream surface | Why not |
|---|---|
| **Declarative workflows** (Power Fx / Copilot Studio actions, formula state, `SensitivityLevel` guards on workflow state) | The upstream schema is an imperative Power Platform DSL that does not map onto this port's graph engine. `agent-framework-declarative` loads agent manifests and a Rust-native `WorkflowSpec` instead, so fixes to the formula engine, expression evaluator or action routing have nowhere to land. |
| **ChatKit** (`chatkit`) | An adapter for OpenAI's ChatKit frontend protocol, which has no Rust server or client ecosystem to serve. |
| **DevUI frontend** | The bundled web UI is a TypeScript app. `agent-framework-hosting` serves the DevUI entity and Responses routes plus its security middleware; the UI-only routes need the frontend's contract and are not ported. |
| **Telegram hosting** (`hosting-telegram`) | Alpha helpers converting one chat platform's update JSON to and from run values. Outside a framework port's remit. |
| **Sandboxed code execution** (`hyperlight`, `monty`, .NET `LocalCodeAct`) | Each wraps a specific sandbox runtime (Hyperlight micro-VMs, the Monty Python interpreter, a local process). There is no Rust binding to the first two worth pinning to, and they change upstream frequently. |
| **TypeSafe AI** (`typesafe`) | An alpha chat-client adapter for one vendor's structured-decision models, which return scores rather than chat text. Niche, and outside the general provider set. |
| **Lab** (`lab`) | Upstream's own incubator for unstable ideas, not a supported surface. |
| **Azure AI Search agentic / Knowledge Base mode** | The connector implements semantic and vector retrieval (including scoped filters). The agentic mode is where upstream moves fastest on this connector and is a documented boundary, not a defect. |
| **Python-runtime specifics** | asyncio task lifetimes, SDK stream disposal, `ResponseStream` latency work, postponed-annotation detection, `py.typed`, pickle allowlists (checkpoints here are JSON), pydantic settings coercion. None has an analogue in this architecture. |
| **.NET hosting plumbing** (ASP.NET Core / Aspire integration, DI extensions, `Workflows.Generators`) | Framework integration specific to the .NET stack. The hosting crate serves the same wire protocols over its own router. |

## Expressed differently

These are covered, just not in upstream's shape. A drift commit that lands
on one of them still needs checking against the Rust form.

| Upstream | Here |
|---|---|
| `run(stream=...)` / `get_response(stream=...)` | Method pairs: `run` / `run_stream`, `get_response` / `get_streaming_response`. One function returning either a value or a stream on a runtime flag is not idiomatic Rust. |
| Ollama's native API | `agent-framework-ollama` speaks Ollama's OpenAI-compatible endpoint and reuses `agent_framework_openai::convert`, so fixes to native fields (`images`, `keep_alive`, native tool-result shapes) do not apply. |
| Anthropic `response_format` handling | Folded into the system prompt (the Messages API has no native field), so upstream's schema-mutation fixes cannot arise. |

## Not yet ported

Open work, recorded so it is not mistaken for a decision. The order comes
from the roadmap, not from this file.

- **Agent hooks, remainder** (`experimental-agent-hooks`): §5.4 result
  labels are unioned from permitting verdicts onto the combined verdict and
  its record, as the agent-hooks SDK (0.1.0-beta.3) does; neither upstream
  nor the SDK persists them with produced data or resurfaces them as
  `extensions.*.source_labels`, so there is nothing further to port.
  FIDES's `source_labels` is a separate, ported feature (`security`).
- **Evaluation, remainder**:
  Evaluation is ported (`experimental-evals`, with `FoundryEvals` behind
  `agent-framework-foundry`'s `evals` feature); upstream's
  `evaluate_foundry_target` is not.
- **Foundry hosting, remainder** (`agent-framework-foundry-hosting`):
  container file citations announced in an MCP tool result's `_meta`; the
  agentserver's resilient task runtime, which runs on the Foundry Tasks
  service and can resume a workflow from its checkpoint (interrupted
  background responses are failed at startup instead); voice invocations;
  and host-side provider-native background polling (depends on the core
  `background`/`continuation_token` contract). Span enrichment stamps agent
  identity at span start (the Rust SDK has no on-end mutation). A toolbox
  skill refresh rebuilds the `SkillsProvider`, so previously loaded skills
  are re-injected only once loaded again. Tested in-process only.
- **Computer use, remainder** (`experimental-computer-use`): Anthropic and
  Gemini skip the computer tool with a warning; upstream has no computer-use
  support for either provider. The workflow has no request-cancellation API,
  so a pending computer call is cancelled from the reply instead
  (`{"call_id", "cancelled": true}` or `{"cancelled": true}`).
- **Durable Task hosting** (`agent-framework-durabletask`, now published
  from its own repository). Blocked: it sits on the `durabletask` SDK, a
  deterministic-replay orchestration engine with durable entities and a
  gRPC sidecar protocol (about 18,000 lines on top of the package's own
  20,000). There is no Rust Durable Task SDK to build on, and no sidecar or
  scheduler emulator to verify a port against.
- **Azure**: the Cosmos DB memory provider (blocked on the Agent Memory
  Toolkit). The Application Insights exporter is ported
  (`agent-framework-core`'s `azure-monitor` feature); the
  `azure_core_opentelemetry` bridge for the Azure SDK's own client spans
  waits on that crate moving to `opentelemetry` 0.32 (see
  [`docs/azure-rust-sdk.md`](docs/azure-rust-sdk.md)). The Blob session
  store is ported (`agent-framework-azure`'s `session-store` feature).
- **Depth gaps the parity ledger surfaced** (see
  [`docs/parity/STATUS.md`](docs/parity/STATUS.md) for every declaration):
  skills beyond inline and file-based skills with scripts (class-based and
  MCP skills, source pipelines); compaction's in-place annotation model (groups here are computed per pass rather than stamped on messages, and summaries are not linked back to the messages they replace); context
  providers seeing the agent object (they see its
  identity, `SessionContext::agent`, and the session in `before_run`, and
  the session after the run, but not the agent itself).
- **Live verification of the database vector stores**: DuckDB is tested
  in-process; the SQL Server (`agent-framework-sql-server`), MongoDB Atlas
  (`agent-framework-mongodb`) and Azure DocumentDB
  (`agent-framework-azure-documentdb`) stores are tested offline (statement
  and pipeline building) with live tests gated on `AF_SQL_SERVER_TEST_URL`,
  `AF_MONGODB_TEST_URL` and `AF_DOCUMENTDB_TEST_URL`, which CI does not
  start. The Redis vector store's live tests likewise need a Redis with
  Search and JSON (`AF_REDIS_SEARCH_TEST_URL`).
- **Adopting `azure_data_cosmos`** (1.0.0 stable since 2026-10) behind an
  optional feature, as `entra-sdk` does for credentials. Roughly 1,540 of
  `agent-framework-cosmos`'s 5,222 lines are the REST plumbing and
  resource-token signing the official client now does. Not a defect — the
  hand-rolled client works — but it is maintenance this port would stop
  carrying. Needs a check that the crate's query surface covers the
  `VectorDistance` queries `vector_store.rs` issues. See
  [`docs/azure-rust-sdk.md`](docs/azure-rust-sdk.md).

- **MCP 2.x, remainder**: the `2026-07-28` era is negotiated
  (`McpProtocolMode`, `server/discover`, per-request `_meta`, the
  `MCP-Protocol-Version` header). Not ported from that era:
  `subscriptions/listen` streams (list-changed notifications still arrive
  only the legacy way), the `input_required` result type (refused with an
  error), result caching hints (`ttlMs`, `cacheScope`) and the Tasks
  extension.
- **Workflow pause/restore bookkeeping** (upstream #9291, #9289):
  preserving concurrent steps across a pause, and restored sub-workflow
  sibling requests. Both land on upstream's `WorkflowExecutor`; this
  runner's superstep and request-info bookkeeping differs enough that
  neither maps mechanically.
