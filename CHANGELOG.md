# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/) (pre-1.0: minor bumps
may break APIs).

## [0.1.0] — 2026-07-13

First release: a Rust implementation of the Microsoft Agent Framework at
structural parity with upstream (`microsoft/agent-framework`) as of upstream
commit `beb65b21` (2026-07-13). 22 crates.

### Core (`agent-framework-core`)

- Data model: `Message`, the `Content` union (text, reasoning, data/URI,
  function call/result, hosted tool calls/results, usage, approvals),
  `ChatResponse`/`AgentResponse` (+ streaming updates and aggregation),
  `ChatOptions` with Python `&`-merge semantics, structured output
  (`ResponseFormat` + `parse_json`), typed `UsageDetails` with
  cache/reasoning counts, **embeddings** (`Embedding`, `GeneratedEmbeddings`,
  `EmbeddingGenerationOptions`, `EmbeddingClient`).
- `ChatClient` trait + `FunctionInvokingChatClient`: parallel tool
  execution, human-in-the-loop approval gating, declaration-only (frontend)
  tools, **progressive tool exposure** (live `FunctionInvocationContext::tools`
  with `add_tools`/`remove_tools`), retry layer (`RetryingChatClient`)
  honoring `Retry-After`.
- Agents: `Agent`/`AgentBuilder`, three-level middleware (agent/chat/
  function), per-run options, dynamic `ToolSource`s (MCP catalog changes),
  `as_tool` (with `propagate_session` child-session semantics,
  `stream_callback`, `approval_mode`).
- Sessions: `AgentSession` + shared-by-reference `SessionState`,
  `HistoryProvider` (in-memory/file), context providers
  (`before_run`/`after_run`), history compaction (four strategies +
  `CompactionProvider`), skills (progressive disclosure), settings
  (`SecretString`, `load_setting`).
- Workflow engine: Pregel-style supersteps, checkpointing (+ resume,
  graph-signature validation), human-in-the-loop pause/resume, output
  designation, async edge conditions, shared state, sub-workflows,
  Mermaid/DOT visualization.
- Orchestrations: Sequential, Concurrent, GroupChat, Handoff (enforced mesh
  topology), Magentic (plan review + stall intervention HITL),
  `WorkflowAgent`, post-agent approval (`AgentApprovalExecutor`).
- Observability: OTel GenAI-semconv spans and (feature-gated) metrics.

### Providers

- OpenAI (Chat Completions + Responses + **embeddings**), Azure OpenAI
  (api-key + Entra ID credential chain, Responses, **embeddings**),
  Anthropic (incl. Bedrock/Vertex/Foundry cloud transports), AWS Bedrock
  (Converse, dependency-free SigV4), Foundry (Responses + Prompt Agents),
  Foundry Local, Gemini, Mistral (chat + **embeddings**), Ollama (chat +
  **embeddings**), GitHub Copilot (token exchange), Copilot Studio.

### Integrations & hosting

- MCP (stdio/HTTP/websocket, sampling, prompts, roots), A2A client + serving,
  declarative agents/workflows (Rust-native `WorkflowSpec`), hosting crate
  (DevUI-style API, AG-UI protocol, OpenAI-compat endpoint, security
  middleware), Redis / Mem0 / Azure AI Search context providers, Cosmos DB
  message store + checkpoint storage, Purview compliance middleware.

### Known divergences from upstream

Documented in `ALIGNMENT_PROGRESS.md` / `PARITY.md`: streaming is expressed
as Rust method pairs (`run`/`run_stream`); the declarative *workflow* DSL is
Rust-native rather than Power Platform; `durabletask`, the `@experimental`
harness/security/evaluation modules, and the Claude Agent SDK wrapper are out
of scope; DevUI's bundled-frontend routes are partial.
