# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/) (pre-1.0: minor bumps
may break APIs).

## [Unreleased]

Upstream-alignment pass against `microsoft/agent-framework` `266206e`
(2026-08-07), covering the six commits that landed after the `4b1afd90`
baseline. Five are .NET-only changes to subsystems this port does not
implement; the sixth established an invariant this port was violating. See
[`ALIGNMENT_PROGRESS.md`](./ALIGNMENT_PROGRESS.md) for the per-commit triage.

### Fixed

- **The tool loop reported only its last iteration's token usage.** Each model
  call in the function-invocation loop reports its own usage, and every exit
  path returned the final call's `ChatResponse` untouched — so a run that
  called tools five times reported roughly a fifth of the tokens it spent, and
  the `gen_ai.usage.*` OTel metrics (which read `usage_details`) under-reported
  with it. Usage is now summed across every iteration and applied to whichever
  response the loop returns, including the approval pause, the
  declaration-only hand-back, and the tools-disabled failsafe. A run where no
  iteration reported usage still reports none rather than a synthesized zero.
  (upstream #7539)

## [0.2.0] — 2026-08-08

Upstream-alignment pass against `microsoft/agent-framework` `4b1afd90`
(2026-08-07), re-baselining from `beb65b21` (2026-07-13). See
[`ALIGNMENT_PROGRESS.md`](./ALIGNMENT_PROGRESS.md) for the full triage of the
112 intervening upstream commits, including the items deliberately left open.

### Fixed

- **Structured output was parsed from the wrong text.** The JSON value was
  built from every message joined with separators and included reasoning
  content, so a tool result could be mistaken for the answer, chain-of-thought
  was folded into the payload, and a JSON document split across streaming text
  chunks had separators injected into it. Now taken from the last non-empty
  assistant message's `text` contents, concatenated with no separator.
  (upstream #6990)
- **Anthropic streaming double-counted tokens.** Anthropic streams cumulative
  usage snapshots and the port summed them as if they were increments, so a
  response reporting 25 input tokens aggregated to 50. (upstream #7162)
- **OpenAI Chat Completions rejected some author names.** A `author_name`
  containing `/`, `|`, `\`, `<` or `>` was sent verbatim and failed the whole
  request with a 400; one containing a space was silently dropped. Names are
  now sanitized to `[a-zA-Z0-9_]` and truncated to 64 characters, matching the
  Python and .NET clients. (upstream #7126)
- **Gemini 3 function-call replays lost `thought_signature`.** Gemini 3
  requires the signature echoed when a call is replayed. Both placements are
  handled: on the function-call part itself (the usual one) and on a preceding
  thought part (backfill only). Reasoning content is no longer sent back as a
  part, matching upstream. (upstream #7095)
- **Approval replacement is now a single ordered walk.** The outstanding-call
  set is derived as the walk decides each content, rather than maintained as
  separate bookkeeping around an order-blind pre-scan — which netted a call
  against a result arriving *after* a replayed request and expanded the
  request into a duplicate declaration.
- **Approval round-trips could duplicate a function call.** The restored call
  was deduped against only the message being scanned, but a hosting layer
  replays the stored call and its approval request as two separate messages, so
  a second copy was restored and left unanswered — which the Responses API
  rejects with "No tool output found for function call ...". (upstream #7271)
- **Compaction could emit conversations providers reject.** A function call and
  its result are now retained or dropped together — previously `TokenBudget`
  could keep a tool result whose call fell outside the budget (a tool message
  answering nothing), and `SelectiveToolResult` deleted stale results while
  their assistant `tool_calls` entries remained (an unanswered call). Either
  half alone is a 400 on the next request. (upstream #7406)
- **Compaction could reduce a conversation to system messages only.**
  `Truncation`/`SlidingWindow` with a budget at or below the system prefix left
  no turn for the model to answer; the most recent non-system message is now
  retained even when that exceeds the limit. (upstream #7219)

### Changed

- **BREAKING: mem0 retrieval scope no longer inherits the storage scope.**
  `Mem0Provider::before_run` searched with the storage `user_id`/`agent_id`, so
  a provider configured with a shared `agent_id` retrieved memories written by
  every user of that agent and injected them into the current user's
  conversation. Retrieval now uses only `with_search_user_id` /
  `with_search_agent_id` / `with_search_application_id`; with none set, nothing
  is retrieved and a warning is logged once. Code that retrieved via
  `with_user_id` alone must add `with_search_user_id`. (upstream #7531)
- **`SelectiveToolResult` now replaces a stale tool result's payload with
  `OMITTED_TOOL_RESULT` instead of deleting the result content.** Deleting it
  orphaned the matching function call; replacing the payload sheds the same
  bulk while keeping the exchange valid. Mirrors the intent of upstream's
  `ToolResultCompactionStrategy`, which replaces stale tool groups with a
  compact stand-in rather than removing them.

### Added

- **Bedrock Converse image blocks.** The Bedrock converter previously dropped
  all `Data`/`Uri` content; inline images in Converse's accepted formats
  (`png`/`jpeg`/`gif`/`webp`) are now emitted as `{"image": ...}` blocks,
  closing a parity gap with upstream's Bedrock client.
- **`Content::renders_on_every_provider`.** The wire-visibility contract that
  compaction's minimum-retention logic depends on now lives on `Content`,
  pinned by contract tests in each provider crate — a converter change that
  invalidates a row fails a test next to the converter instead of surfacing as
  a compaction bug. `Uri` content is excluded (Bedrock has no remote-URL image
  source), and `Data` images are bounded by Bedrock's format set.
- `DataContent::from_uri` / `DataContent::media_type_from_uri`: validating
  construction from a `data:` URI, rejecting a missing scheme, missing `,`, or
  non-base64 declaration instead of silently mis-slicing it. (upstream #6916)
- `TextReasoningContent::protected_data`, mirroring upstream's
  `Content.protected_data`, for provider-opaque reasoning replay tokens.
- OpenAI cache-**write** token counts on both the Chat Completions and
  Responses surfaces, populating `UsageDetails::cache_creation_input_token_count`
  and so the `gen_ai.usage.cache_creation.input_tokens` OTel attribute.
  (upstream #7369)

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
