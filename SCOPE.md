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

- **Agent hooks, remainder** (`experimental-agent-hooks` ships the
  enforcement bundle): the approval seam (liftable denies are enforced as
  plain denies), the `sequential/run_all` and parallel composition profiles,
  the `jcs-sha256` and custom identity providers (records are
  identity-unbound), an `agent_shutdown` on cancellation, and §5.4
  result-label flow (labels reach the combined verdict and the record but
  are not persisted with the produced data or resurfaced as
  `source_labels`; upstream's framework layer does not do this either).
- **Security (FIDES)**: marked experimental upstream; when ported it goes
  behind an `experimental-*` feature (see `docs/feature-stages.md`).
  Evaluation is ported (`experimental-evals`, with `FoundryEvals` behind
  `agent-framework-foundry`'s `evals` feature); upstream's
  `evaluate_foundry_target` is not.
- **Foundry hosting** (`foundry_hosting`, `Microsoft.Agents.AI.Foundry.Hosting`).
- **Durable Task hosting** (`durabletask`), blocked on the sidecar protocol.
- **Claude Agent SDK agent** (`claude`), a subprocess shim with no Rust SDK
  to build on.
- **Azure**: the Cosmos DB memory provider (blocked on the Agent Memory
  Toolkit). The Application Insights exporter is ported
  (`agent-framework-core`'s `azure-monitor` feature); the
  `azure_core_opentelemetry` bridge for the Azure SDK's own client spans
  waits on that crate moving to `opentelemetry` 0.32 (see
  [`docs/azure-rust-sdk.md`](docs/azure-rust-sdk.md)). The Blob session
  store is ported (`agent-framework-azure`'s `session-store` feature).
- **Depth gaps the parity ledger surfaced** (see
  [`docs/parity/STATUS.md`](docs/parity/STATUS.md) for every declaration):
  skills beyond inline and file-based skills (scripts and script
  runners, class-based and MCP skills, source pipelines); compaction's message-group model, triggers and summarization;
  workflow telemetry, run cancellation and graph reflection; context
  providers seeing the agent object (they see its
  identity, `SessionContext::agent`, and the session in `before_run`, and
  the session after the run, but not the agent itself).
- **Vector stores**: MongoDB, DuckDB, SQL Server and Azure DocumentDB
  (blocked on the MongoDB wire protocol). The Redis vector store is ported
  (`agent-framework-redis`'s `experimental-vector-stores` feature); its
  live tests run only against a Redis with Search and JSON
  (`AF_REDIS_SEARCH_TEST_URL`), which this repository's CI does not start.
- **Adopting `azure_data_cosmos`** (1.0.0 stable since 2026-10) behind an
  optional feature, as `entra-sdk` does for credentials. Roughly 1,540 of
  `agent-framework-cosmos`'s 5,222 lines are the REST plumbing and
  resource-token signing the official client now does. Not a defect — the
  hand-rolled client works — but it is maintenance this port would stop
  carrying. Needs a check that the crate's query surface covers the
  `VectorDistance` queries `vector_store.rs` issues. See
  [`docs/azure-rust-sdk.md`](docs/azure-rust-sdk.md).

- **MCP 2.x protocol negotiation** (upstream #9236): the `2026-07-28`
  protocol era upstream's `auto` mode negotiates. This client requests
  `2025-06-18` and accepts servers that answer with `2025-11-25` or older;
  the new era's wire changes are not ported.
- **Workflow pause/restore bookkeeping** (upstream #9291, #9289):
  preserving concurrent steps across a pause, and restored sub-workflow
  sibling requests. Both land on upstream's `WorkflowExecutor`; this
  runner's superstep and request-info bookkeeping differs enough that
  neither maps mechanically.
