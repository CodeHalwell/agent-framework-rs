# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/) (pre-1.0: minor bumps
may break APIs).

## [Unreleased]

### Added

- **Entra ID credentials from the official Azure SDK for Rust**, behind
  `agent-framework-azure`'s new optional `entra-sdk` feature.
  `SdkTokenCredential` adapts any [`azure_identity`] credential onto this
  crate's `TokenCredential`, so SDK and hand-rolled credentials are
  interchangeable wherever a credential is accepted. This is additive: the
  hand-rolled chain remains the default and is unchanged when the feature is
  off. It exists to reach credential types this crate does not implement —
  `ClientCertificateCredential`, `ClientAssertionCredential`,
  `AzurePipelinesCredential`, `AzureDeveloperCliCredential` — and to let
  Microsoft own IMDS quirks, sovereign-cloud endpoints and token lifetimes
  under a semver guarantee.
  - `azure_core`/`azure_identity` are pinned with `default-features = false`
    deliberately. Their defaults pull `aws-lc-rs` (a C library requiring
    `cmake`) as a *second* `rustls` crypto backend beside the `ring` one
    `reqwest` already uses here, which leaves `rustls` without an unambiguous
    process-level default provider, plus a platform-verifier/Android-JNI
    stack. With defaults off and `azure_core`'s `reqwest` feature on, it
    reuses the existing reqwest + ring TLS; the workspace lock gains only
    pure-Rust crates and no C toolchain requirement.

- **A ready-made OTLP export pipeline**, behind `agent-framework-core`'s new
  optional `otel-export` feature (re-exported by the umbrella crate).
  `observability::export::OtelExport` builds an OTLP exporter, tracer
  provider, meter provider and the `tracing`↔OpenTelemetry bridge in one call,
  wired to the GenAI conventions this crate already emits. Enabling it implies
  `otel-metrics`, since the pipeline installs the `MeterProvider` those
  histograms record through. This is the only feature that pulls an OTel
  *SDK* — the default build stays API-only, so consumers who don't ask for
  export never inherit the SDK or its version churn.
  - `OtelPipeline::install` claims the global subscriber for applications that
    have none; `OtelPipeline::tracing_layer` returns the bridge layer for
    applications composing their own.
  - Transport is OTLP-over-HTTP/protobuf on reqwest's *blocking* client, not
    gRPC and not the async client. gRPC would add a second HTTP stack; the
    async client panics with "there is no reactor running" when the batch span
    processor flushes, because that processor and the periodic metric reader
    each export from a background thread with no tokio runtime on it.
  - `examples/observability/otel_export.rs` demonstrates both routes and is
    compiled by CI, so the wiring cannot drift — replacing the previous
    ` ```ignore ` snippet in the module docs, which was never compiled.

### Fixed

- **`observability`: corrected stale documentation on the third GenAI
  histogram.** The module docs and
  `metrics::record_function_invocation_duration` both stated that
  `agent_framework.function.invocation.duration` was defined but "not yet
  called anywhere in this crate", and listed switching to `tool_span_ex` and
  adding `record_tool_arguments`/`record_tool_result` as outstanding
  follow-ups. All of that had in fact landed —
  `FunctionInvokingChatClient` times each tool invocation and records the
  histogram, and uses all three span helpers. A reader following the old docs
  would have concluded tool-call timing was unavailable, or wired up a second
  recording of it. Documentation only; no behavior change.

## [0.3.0] — 2026-08-24

Upstream-alignment passes against `microsoft/agent-framework`, moving the
baseline from `4b1afd90` (2026-08-07) to `a63d462` (2026-08-24). Six
upstream changes are ported and 100+ intervening commits are triaged in
[`ALIGNMENT_PROGRESS.md`](./ALIGNMENT_PROGRESS.md), which records why each
one that does not apply does not.

This release **breaks API** (pre-1.0, so a minor bump): the observability
recording functions take an `&ObservabilityConfig` in place of a
`capture_content: bool`, `chat_span` takes the semantic-convention flag, and
`gen_ai.system` is no longer emitted alongside `gen_ai.provider.name`. See
**Changed** below.

### Changed

- **The GenAI semantic-convention version is now selectable, and the provider
  tag follows it** (upstream #7673, [BREAKING] there). `gen_ai.system` was
  renamed to `gen_ai.provider.name` above the OTel v1.36.0 baseline, and this
  port emitted *both* names on every chat span, so a consumer pinned to the
  baseline saw an attribute its version does not define. `ObservabilityConfig`
  now reads `OTEL_SEMCONV_STABILITY_OPT_IN` and exposes
  `use_latest_experimental_gen_ai_semconv()`: unset means the latest
  conventions (upstream's default too), and a list omitting
  `gen_ai_latest_experimental` selects the baseline. Exactly one provider
  attribute is emitted — on spans and on the metrics attributes — and the four
  above-baseline attributes (`gen_ai.usage.cache_creation.input_tokens`,
  `gen_ai.usage.cache_read.input_tokens`,
  `gen_ai.usage.reasoning.output_tokens`, `gen_ai.tool.definitions`) plus
  `gen_ai.tool.call.arguments` / `gen_ai.tool.call.result` are withheld at the
  baseline. Under the default nothing changes except that `gen_ai.system` is
  no longer emitted alongside `gen_ai.provider.name`.
  - API: `record_request`, `record_response`, `record_tool_arguments` and
    `record_tool_result` take `&ObservabilityConfig` in place of a
    `capture_content: bool`; `chat_span` takes the semconv flag;
    `ObservableChatClient` gained `with_observability_config`
    (`with_content_capture` still works and now sets the flag on the config).

### Added

- **`Error::MiddlewareFailure`, a fail-closed signal for function middleware**
  (upstream #7562). The function-invocation loop absorbs every error a tool or
  its middleware produces into a tool-error result, hands it to the model and
  keeps looping — the right default for a tool failure the model can route
  around, but it left an enforcement layer (a guardrail, a policy or
  authorization gate) no way to stop a run: refusing a call just produced an
  error string the model could try again. Middleware returning
  `Error::middleware_failure(..)` is now propagated instead of absorbed, and
  because the parallel batch is driven by `try_join_all`, propagating it also
  drops the sibling calls still in flight. Every other error keeps the
  absorb-and-continue contract unchanged.

### Fixed

- **Replaying a conversation duplicated stored history** (upstream #7242). A
  history provider is handed a run's input plus its response, so a caller that
  keeps its own transcript and replays all of it each turn handed back
  everything already stored — and each provider appended it unconditionally.
  History grew superlinearly, and since `before_run` prepends it to the
  request, the duplicated turns were resent to the model on every later run.
  `agent_framework_core::history::filter_new_messages` now aligns the stored
  run inside the incoming one (by `message_id`, or by role and contents when
  there is none) and stores only what follows it; `InMemoryHistoryProvider`,
  `FileHistoryProvider`, `RedisChatMessageStore` and `CosmosChatMessageStore`
  all use it, the last two reading their stored history first (and skipping
  that read entirely when there is nothing to store, or when a Redis store is
  configured to retain nothing). A run that cannot be aligned is appended
  exactly as before — unlike upstream, no set-based fallback drops a turn that
  merely repeats an earlier one. Alignment sees a run's **input** only:
  response messages were just generated and can never be a replay, so they are
  always stored, even when one happens to reproduce the stored tail.
- **A replayed transcript was also sent to the model twice.** Storing only the
  new suffix fixed history growth, but the request is assembled the other way
  round — injected context first, then the caller's input — so a history
  provider that unconditionally injected what it held sent `q1, a1, q1, a1,
  q2` for a caller replaying `q1, a1, q2`. All four history providers now
  inject nothing when the run's input already carries the stored run
  (`inject_stored_history`).
  `StoredHistory::{Complete, Window}` selects where a stored run is looked for:
  a complete history is matched at the start only — it begins at the
  conversation's first message, so a replay of it can only begin with it — while
  a window is searched for, preferring a match at the start (an at-cap list that
  has never actually been trimmed is still complete) and otherwise taking the
  last occurrence. The Redis store asks for `Window` only when its list is at
  its cap. Alignment also requires *evidence* that stored history could be a
  replay at all — a matching message id, or a non-user turn, since a replay is
  a transcript and carries the assistant's replies. Stored history that is
  nothing but id-less user messages is indistinguishable from new input that
  repeats it, and is left alone. An empty `message_id` counts as no id at all,
  matching the `!id.is_empty()` guard the crate already applies elsewhere.
- **Tool spans could report a different semconv version than chat spans.** The
  function-invocation loop rebuilt an `ObservabilityConfig` from the
  environment for every tool call, so a client configured explicitly for one
  convention version emitted tool spans under whatever the environment said.
  `FunctionInvokingChatClient` now carries the config, settable with
  `with_observability_config` and resolved from the environment once at
  construction. `AgentBuilder::observability_config` reaches that wrapper,
  which the builder constructs itself.
- **A Redis retention limit of zero retained everything** (upstream #7470).
  `RedisChatMessageStore::with_max_messages(0)` is a request to retain
  nothing — unlimited is expressed by not calling it at all — but trimming to
  `-(max)` emits `LTRIM key 0 -1` for a limit of zero, which is Redis's "keep
  the whole list", so the trim ran on every save and did nothing.
  `add_messages` now returns before serializing, so no payload reaches Redis
  (or an AOF or a replica) even briefly. Stored history is deliberately left
  alone rather than deleted: the key carries no per-provider discriminator, so
  two stores sharing a prefix and session id address the same list, and
  deleting it would drop a co-located store's history. Use `clear` to remove
  history. A negative limit — the other half of upstream's fix — is
  unrepresentable here, since `max_messages` is a `usize`.
- **Gemini 3 thought signatures were dropped across an approval round trip**
  (upstream #7546). Gemini 3 rejects a `functionCall` part that lacks the
  `thoughtSignature` it was issued with. Signatures were paired to calls by
  adjacency alone, and any intervening content cleared the held signature —
  so a `FunctionApprovalResponse` sitting between a reasoning carrier and its
  call dropped it, and a call replayed in a later message could never be
  signed at all. Both turns then failed with a 400. Content that emits no
  wire Part no longer clears the signature, and a `call_id -> signature` map
  accumulated as the conversation is emitted signs a later replay. Precedence
  is unchanged: the call's own `protected_data`, then an adjacent carrier,
  then the map. The map is written by the emit walk rather than a pre-pass, so
  the pairing rules have a single implementation — a pre-pass that restated
  them laxly would re-sign the very calls adjacency had refused.
- **Stateless Responses requests never asked for the encrypted reasoning
  item.** A reasoning item is only replayable on the next turn of a `store:
  false` tool loop if it carries `encrypted_content`, and the service only
  returns that when the request's `include` asks for it. This port set
  `include` nowhere, so the replay path in `messages_to_input` — which
  re-emits the item verbatim and drops one lacking `id`/`encrypted_content` —
  had nothing valid to re-send. Both Responses clients (OpenAI and Azure
  OpenAI) now add `reasoning.encrypted_content` when a request carries no
  service-side-storage indicator, matching upstream. A caller's own `include`
  entries are preserved and never duplicated.
- **Foundry opts out of the above**, matching upstream #7536: it does not want
  encrypted reasoning unless asked for by name. New
  `AzureOpenAIResponsesClient::without_implicit_encrypted_reasoning`, which
  `FoundryChatClient` sets on its transport. An explicitly requested
  `reasoning.encrypted_content` is still honored.
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
