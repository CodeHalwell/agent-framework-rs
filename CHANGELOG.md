# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/) (pre-1.0: minor bumps
may break APIs).

## [Unreleased]

Upstream parity sync. 0.1.1 was cut against `microsoft/agent-framework`
`beb65b21` (2026-07-13); this re-pins the port to **`c6442de`
(2026-07-24)**. All 102 upstream commits in that window were triaged and
nothing is left outstanding: every change is ported, already at parity, or
not applicable, each with its reasoning recorded in
[UPSTREAM_SYNC.md](UPSTREAM_SYNC.md).

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

- **`.text()` no longer includes reasoning.** `Message::text`,
  `ChatResponseUpdate::text_content` and `AgentResponseUpdate::text` returned
  a model's chain-of-thought concatenated with its answer, because
  `Content::as_text` matches reasoning as well as text. They now use the new
  `Content::as_plain_text`, matching upstream's `content.type == "text"`
  filter on all five of its `.text` properties. `as_text` stays inclusive for
  its one remaining caller, compaction's token counting, where a reasoning
  block genuinely costs tokens.
- **OpenAI image `detail` is no longer dropped.** It is read from the image
  content's `additional_properties`, as upstream does, and forwarded on the
  `image_url` part.
- **Sub-workflow state survives a parent checkpoint/resume.** A parent's
  checkpoint recorded nothing about its sub-workflows, so a resumed parent met
  a child that had forgotten its executor state, its in-flight messages and
  its place in the superstep loop — and a response to a forwarded request had
  no run to route back into. `WorkflowExecutor` now embeds each paused child
  run's own checkpoint in its executor state and rebuilds it on restore. A
  child whose graph no longer matches its checkpoint is dropped with a warning
  rather than failing the whole parent restore (upstream #7097).
- **A sub-workflow's own checkpoint storage is detached.** A sub-workflow is
  checkpointed by its parent; its own storage wrote a second, independent
  series of checkpoints that nothing ever resumed from (upstream #7097).

### Changed

- `gen_ai.response.finish_reasons` is now only recorded for the four values
  the OpenTelemetry GenAI convention defines, and `tool_calls` is emitted
  under the convention's name `tool_call`. `FinishReason` is an open string
  enum, so provider-specific values previously landed on spans verbatim
  (upstream #7105).

### Added

- **Context-message attribution.** `SessionContext::extend_messages` stamps
  each injected context message with the provider that contributed it, and
  `extend_messages_from_sessions` additionally records the sessions the
  content came from under
  `additional_properties["_attribution"]["origin_session_ids"]`, so
  downstream observers can tell cross-session content apart for governance or
  audit. Origins accumulate across providers; the other attribution keys are
  first-writer-wins (upstream #7041).
- **MCP tool-use sampling results.** When a sampled model asks for tools, the
  `sampling/createMessage` reply is now MCP's tool-result shape — an array of
  `tool_use` blocks with `stopReason: "toolUse"` — instead of being scanned
  only for text and erroring out as having nothing to return. The scan also
  spans every message in the response rather than only the first (upstream
  #7189).
- **OpenAI prompt cache breakpoints** for GPT-5.6 and later. Set
  `prompt_cache_breakpoint` in a content item's `additional_properties` to
  mark a cache breakpoint on its request part (text, image, audio or file);
  pair it with a request-wide `prompt_cache_options` via
  `ChatOptions::additional_properties`, which is forwarded verbatim. A
  message carrying a breakpoint is emitted in typed content-part form even
  when it is text-only, since a plain-string `content` has nowhere to carry
  one (upstream #7163).
- **Responses session continuity in the hosting crate.**
  `ResponsesRequest` gained `previous_response_id` / `conversation_id` and a
  `session_id()` helper returning a `SessionId` — `PreviousResponse` or
  `Conversation`, since only a conversation id is echoed back to the client.
  An id without its conventional `resp_` / `conv_` prefix is accepted but
  warns. `ResponseObject::with_conversation` renders the `conversation` field,
  and `hosting::conversation_id()` mints `conv_…` ids. DevUI's non-streaming
  agent route is wired through it (upstream #7234).
- **Nested workflow checkpoint primitives.**
  `WorkflowRun::capture_checkpoint_object` snapshots a quiescent run without
  persisting it, and `Workflow::restore_run_from_checkpoint_object` rebuilds a
  paused run without resuming it, so a caller decides when it runs again.
  These back the sub-workflow fix above and are public for anyone embedding a
  workflow's state elsewhere (upstream #7097).
- `types::structured_output_text(&[Message]) -> String` — the
  structured-output text-extraction helper described above.
- `Content::as_plain_text()` — text content only, excluding reasoning.
- `TextReasoningContent::protected_data: Option<String>` — opaque
  provider-signed data for a reasoning step that must be echoed back on the
  next turn, mirroring upstream's `Content.protected_data`. Additive and
  serde-optional, so existing serialized content deserializes unchanged.
- `additional_properties` on `TextContent`, `DataContent` and `UriContent` —
  provider-specific per-content extras, mirroring upstream's
  `Content.additional_properties`. Additive and serde-optional.
  `DataContent` and `UriContent` now also derive `Default`.

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
