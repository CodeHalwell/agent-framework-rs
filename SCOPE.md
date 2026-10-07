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

- **Harness, remainder**: file access, file memory, memory, background
  agents (and the loop's `background_tasks_running` condition),
  `TodoFileStore` and the `create_harness_agent` assembly. The agent loop,
  todo, tool approval and agent mode are ported behind
  `experimental-harness`.
- **Evaluation** and **security (FIDES)**: both marked experimental
  upstream; when ported they go behind `experimental-*` features (see
  `docs/feature-stages.md`).
- **AG-UI client** (the hosting crate has the server side only).
- **Local shell tool** (hosted shell content types exist).
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
  on the Agent Memory Toolkit), the dedicated Application Insights exporter.
- **Depth gaps the parity ledger surfaced** (see
  [`docs/parity/STATUS.md`](docs/parity/STATUS.md) for every declaration):
  skills beyond inline skills (scripts, file-based sources, source
  pipelines); compaction's message-group model, triggers and summarization;
  workflow telemetry, run cancellation and graph reflection; message source
  attribution and message injection; context providers seeing the agent and
  session.
- **Vector stores**: Postgres, Qdrant, MongoDB, DuckDB, SQL Server, Azure
  DocumentDB (blocked on the MongoDB wire protocol), Valkey.
