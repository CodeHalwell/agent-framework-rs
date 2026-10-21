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

- **Harness, remainder**: the file access tools (`FileAccessProvider`),
  memory, background agents (and the loop's `background_tasks_running`
  condition), `TodoFileStore` and the `create_harness_agent` assembly. The
  agent loop, todo, tool approval, agent mode, the file store and file
  memory are ported behind `experimental-harness`.
- **Agent hooks, remainder** (`experimental-agent-hooks` ships the
  enforcement bundle): the approval seam (liftable denies are enforced as
  plain denies), the `sequential/run_all` and parallel composition profiles,
  the `jcs-sha256` and custom identity providers (records are
  identity-unbound), an `agent_shutdown` on cancellation, and §5.4
  result-label flow (labels reach the combined verdict and the record but
  are not persisted with the produced data or resurfaced as
  `source_labels`; upstream's framework layer does not do this either).
- **Evaluation** and **security (FIDES)**: both marked experimental
  upstream; when ported they go behind `experimental-*` features (see
  `docs/feature-stages.md`).
- **Foundry hosting** (`foundry_hosting`, `Microsoft.Agents.AI.Foundry.Hosting`).
- **Foundry server-hosted agents**: `FoundryAgent` realises a Prompt Agent
  client-side and cannot yet bind to an agent hosted on the Foundry control
  plane (`AIProjectClient.AsAIAgent`, `FoundryAgent(Uri, ...)`).
- **Cloud-transport streaming for Anthropic on Bedrock and Vertex**: the
  streaming methods buffer one non-streaming request into a single update.
  The AWS event-stream and `:streamRawPredict` framings are not implemented.
- **Durable Task hosting** (`durabletask`), blocked on the sidecar protocol.
- **Claude Agent SDK agent** (`claude`), a subprocess shim with no Rust SDK
  to build on.
- **Azure**: Content Understanding, the Cosmos DB memory provider (blocked
  on the Agent Memory Toolkit), the dedicated Application Insights exporter,
  and Cosmos chat history decoupled from its partition-key strategy
  (upstream #9214: tenant/user metadata persisted whatever the strategy, plus
  a caller-chosen partition key). `CosmosChatMessageStore` partitions by
  `threadId` alone and carries no tenant/user fields.
- **A framework `User-Agent` outside the Azure crates**. The Azure-facing
  crates send `agent-framework-rust/<version> (<component>)` via
  `agent_framework_azure::azure_http_client`; the other provider clients, and
  `agent-framework-purview` and `agent-framework-foundry-local`, still build
  bare `reqwest` clients. A framework-wide header needs the helper somewhere
  every crate can reach, and `agent-framework-core` carries no `reqwest`
  dependency by design.
- **Function-invocation pauses with mixed batches**. A declaration-only
  call's local siblings run on the same turn. Upstream instead holds them
  until the caller answers whenever a session is present (#9050). An
  approval batch also defers a declaration-only sibling as an approval,
  where upstream pauses it as Host input (#9203).
- **Streamed code-interpreter content**. The OpenAI Responses client
  produces no `CodeInterpreterToolCallContent`, so there is nothing to
  stream or merge.
- **Per-agent handoff autonomous turn limits**. Upstream's
  `with_autonomous_mode(agents=[...], turn_limits={...})` is per participant.
  Here there is one `HandoffBuilder::max_iterations` cap plus a global
  `HandoffInteractionMode`, whose `Autonomous` variant already expresses
  upstream's zero-continuation case.
- **Depth gaps the parity ledger surfaced** (see
  [`docs/parity/STATUS.md`](docs/parity/STATUS.md) for every declaration):
  skills beyond inline skills (scripts, file-based sources, source
  pipelines); compaction's message-group model, triggers and summarization;
  workflow telemetry, run cancellation and graph reflection; message source
  attribution and message injection; context providers seeing the agent and
  session.
- **Vector stores**: Postgres, Qdrant, MongoDB, DuckDB, SQL Server, Azure
  DocumentDB (blocked on the MongoDB wire protocol), Valkey.
- **MCP 2.x protocol negotiation** (upstream #9236). This port speaks the
  revision it was written against.
- **`SequentialBuilder::ensure_trailing_user_turn`** (upstream #8779): an
  opt-in that appends a synthetic user continuation message when the
  conversation handed to the next participant ends on an assistant turn,
  for providers that answer an assistant-ending prompt with empty text.
- **Workflow pause/restore bookkeeping** (upstream #9291, #9289):
  preserving concurrent steps across a pause, and restored sub-workflow
  sibling requests. Both land on upstream's `WorkflowExecutor`; this
  runner's superstep and request-info bookkeeping differs enough that
  neither maps mechanically.
