# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/) (pre-1.0: minor bumps
may break APIs).

## [Unreleased]

Upstream parity sync. 0.1.1 was cut against `microsoft/agent-framework`
`beb65b21` (2026-07-13); this re-pins the port to **`c6442de`
(2026-07-24)**. All 102 upstream commits in that window were triaged — see
[UPSTREAM_SYNC.md](UPSTREAM_SYNC.md) for the full disposition, including
what was already at parity and what is deliberately deferred.

### Fixed

- **Structured output no longer corrupts JSON split across streaming
  chunks.** `ChatResponse::parse_json`, `AgentResponse::parse_json` and the
  `value` auto-fill read the text of the *last non-empty assistant message*,
  concatenated with no separator. They previously used `text()`, which
  space-joins content items and newline-joins messages: a payload streamed
  as `{"na` + `me":"x"}` became `{"na me":"x"}` — valid JSON with the wrong
  key, so it failed silently rather than loudly. Reasoning content is now
  also excluded, so chain-of-thought can no longer leak into a parsed value
  (upstream #6990). See `types::structured_output_text`.
- **Approval round trips no longer duplicate a function call.** The
  duplicate check in the approval-replay rewrite is conversation-wide
  instead of per-message. A function call and its approval request routinely
  arrive in *separate* messages, so the per-message check restored the same
  `call_id` twice, leaving one copy permanently unanswered. Call ids that
  already carry a real result are excluded from the pending set, so reusing
  a `call_id` for a later invocation still works (upstream #7267).
- **OpenAI Responses finish reasons follow the upstream precedence.**
  `incomplete_details.reason` now outranks a partial `function_call` output
  item (a response truncated mid-tool-call reports `length`, not
  `tool_calls`), and a non-terminal response — the background /
  continuation-token path — reports no finish reason at all instead of
  surfacing `in_progress` / `queued` as if they were ones. The
  `response.incomplete` streaming event is now terminal, so a truncated
  stream carries its finish reason and usage. Azure OpenAI and Foundry
  inherit this by delegation (upstream #7105).
- **Gemini 3 function-call replays keep their thought signature.**
  `thoughtSignature` is parsed off a reasoning part into the new
  `TextReasoningContent::protected_data` and stamped back onto the function
  call that reasoning precedes. Reasoning is no longer replayed as a
  `thought` part of its own. Without the signature Gemini 3 rejects the
  follow-up turn of a tool-calling exchange (upstream #7095).

### Changed

- `gen_ai.response.finish_reasons` is now only recorded for the four values
  the OpenTelemetry GenAI convention defines, and `tool_calls` is emitted
  under the convention's name `tool_call`. `FinishReason` is an open string
  enum, so provider-specific values previously landed on spans verbatim
  (upstream #7105).

### Added

- `types::structured_output_text(&[Message]) -> String` — the
  structured-output text-extraction helper described above.
- `TextReasoningContent::protected_data: Option<String>` — opaque
  provider-signed data for a reasoning step that must be echoed back on the
  next turn, mirroring upstream's `Content.protected_data`. Additive and
  serde-optional, so existing serialized content deserializes unchanged.

## [0.1.1] — 2026-07-13

First published release on crates.io — identical in content to 0.1.0.

- The v0.1.0 release-pipeline run failed at the publish step (the crates.io
  token secret was misnamed), after the `v0.1.0` tag had already been
  pushed, so 0.1.0 was never published and its version number is burned.
- Release workflow: publish to crates.io **before** tagging and creating
  the GitHub Release, so a failed publish no longer burns the version —
  the run can simply be retried after fixing the cause.

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
