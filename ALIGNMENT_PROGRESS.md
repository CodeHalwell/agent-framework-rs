# Alignment progress against current upstream

Tracks the re-baselining of `agent-framework-rs` onto current upstream, as
catalogued in [`UPSTREAM_DRIFT.md`](./UPSTREAM_DRIFT.md). Section numbers under
the `68136ee` heading refer to that document. Every item recorded as landed was
independently verified (full workspace build + `cargo test` + clippy
`--all-targets` + rustfmt, all green) before commit.

**Current upstream baseline: `4b1afd90` (2026-08-07).** Sections are newest
first; each records the upstream revision it was checked against.

## Post-`beb65b21` drift (checked against `4b1afd90`, 2026-08-07)

Upstream moved **112 commits touching `python/packages`** in the ~3.5 weeks
between `beb65b21` (2026-07-13) and `4b1afd90` (2026-08-07). Each was triaged
against the subsystems this port actually implements. The outcome splits four
ways; the counts are the honest picture, not a claim of completeness.

### Ported this pass (6, each with regression tests)

| Upstream | Change | Rust site |
|---|---|---|
| #6990 | **Structured-output text selection.** The JSON value was parsed from *every* message's text, joined with `"\n"`/`" "`, including reasoning content. Three separate corruptions: a tool result or user echo carrying JSON could be mistaken for the answer; a reasoning model's chain-of-thought was prepended to the payload; and a JSON document split across text chunks had separators injected into it (`{"na` + `me":1}` → `{"na me":1}`). Now mirrors upstream's `_last_non_empty_assistant_message_text` — last non-empty **assistant** message, `text` contents only, joined with no separator. | `core/types/response.rs` (`structured_output_text`, used by both `parse_json`s and `try_parse_value`) |
| #6916 | **Data-URI validation.** No validating constructor existed; a malformed `data:` URI was silently mis-sliced or ignored. Added `DataContent::from_uri` / `media_type_from_uri`, rejecting a missing `data:` scheme, a missing `,`, or a non-`;base64` declaration. | `core/types/content.rs` |
| #7126/#7127 | **Chat Completions `author_name` sanitization.** The API validates `name` against `^[^\s<\|\\/>]+$`. The port only skipped names containing whitespace, so `"a/b"` was sent and 400'd the entire request, while `"My Agent"` was dropped rather than sanitized. Now mirrors upstream/.NET `SanitizeAuthorName`: strip outside `[a-zA-Z0-9_]`, omit when empty, truncate to 64. | `openai/convert.rs` |
| #7369 | **OpenAI cache-*write* tokens.** Only cache *reads* were parsed. Added `cache_write_tokens` on both surfaces, populating the typed `cache_creation_input_token_count` (which is what this port's OTel layer reads, so the `gen_ai.usage.cache_creation.input_tokens` attribute now flows without the extra key-mapping table upstream needs). | `openai/convert.rs`, `openai/responses.rs` |
| #7162 | **Anthropic streaming token double-count.** Anthropic streams *cumulative* usage snapshots: `message_delta` repeats the input/cache counts `message_start` already reported. Because `absorb_update` sums every usage content, a stream reporting 25 input tokens aggregated to 50. Added a per-stream `StreamUsageAccumulator` threaded through `SseState` that emits increments. | `anthropic/convert.rs`, `anthropic/lib.rs` |
| #7095 | **Gemini 3 `thought_signature` replay.** Not supported at all. Gemini pairs a thought part with a `thoughtSignature` that must be echoed on the function call that reasoning produced, or the replayed turn is rejected. Added `TextReasoningContent::protected_data` (mirroring upstream's `Content.protected_data`), captured it on parse, and — matching upstream — stopped sending reasoning back as a part, instead carrying its signature onto the immediately following function call. | `core/types/content.rs`, `gemini/convert.rs` |

Verified: full workspace build, `cargo test --workspace --all-features`
(**1508 passing**, 24 of them new), `cargo clippy --all-targets --all-features`
clean, `cargo fmt --check` clean.

### Verified already satisfied — no action (the port was level or ahead)

- **#6809** function-call name lost when a streaming delta carries it late —
  `FunctionCallContent::merge` already fills an empty name from `other`.
- **#7488** Gemini thought summaries surfaced as reasoning content — already
  done; upstream was catching up to the port here.
- **#7292** OpenAI Responses native `instructions` — the port already sends the
  top-level field rather than prepending a system message.
- **#7060** per-run `additional_beta_flags` leaking into the Anthropic request
  body — `compute_beta_flags` already removes the key from
  `additional_properties`.
- **#7105** finish-reason normalization — `map_stop_reason` already maps
  `guardrail_intervened` → `content_filter` and passes unknown reasons through.

### Not applicable — architectural divergence

- **#6822** (Ollama parallel tool calls colliding on `call_id`) and the Ollama
  half of **#7105** (`done_reason` normalization): this port's Ollama client
  targets Ollama's **OpenAI-compatible** `/v1/chat/completions` surface, which
  returns real per-call ids and OpenAI-shaped `finish_reason`s. Upstream's bugs
  live in the native `/api/chat` path, which the port does not have.
- The **AG-UI** cluster (~10 commits), **harness/skills/evaluation**
  graduations, the **durabletask / Azure Functions** extraction,
  **foundry-hosting**, and the **telegram / chatkit / monty / hyperlight / lab**
  packages: no Rust counterpart, already documented under "Remaining" below.

### Triaged as relevant but NOT yet ported

Carried forward as the next pass's work — none is closed, and the list is
roughly in descending value order:

- **Compaction cluster** (#7396, #7391, #7406, #7219, #7124, #7375): bound
  tool-result summaries, ignore excluded tool results, keep call/result
  occurrences atomic, suppress empty projections, fix token counts inflating on
  non-ASCII text, bound summarization input before the provider call.
- **Approvals cluster** (#7462, #7407, #7408, #7410, #7345, #7271, #7090):
  orphaned local approval responses, decisions preserved under OpenAI
  continuation, tool content returned after invocation limits, provider-injected
  approvals deferred to in-run execution, resume/replay, duplicate call on
  round-trip, auto-approval name-collision warnings.
- **Workflow checkpointing**: full replayability (#7374, BREAKING), sub-workflow
  restore preserving sub-workflow state (#7097), checkpoint encoding (#6579).
- **Sessions**: `SessionStore` moved into core + Foundry Responses session
  persistence (#7306), cross-session origin attribution (#7041), hosted session
  snapshot isolation (#7141).
- **Core**: agent-hooks interception contract (#7515, new experimental
  feature), declaration-only streaming metadata (#7409), stateless replay of
  reasoning-paired tool calls (#7233), `from_dict` type enforcement (#7256),
  `PropertySchema` nested recursion (#7200), feature-usage User-Agent telemetry
  (#7420), tool-def JSON for observability (#7029), restricting an unknown
  `finish_reason` from the OTel attribute (#7105).
- **Orchestration**: Magentic manager duplicating conversation history (#6297).
- **Providers**: OpenAI raw JSON-Schema `response_format` passthrough (#7199),
  GPT-5.6 prompt-cache breakpoints (#7163), Responses conversation-ID helper
  (#7234, BREAKING), Foundry agent inheriting `OPENAI_CHAT_MODEL` (#7283),
  GitHub Copilot options forwarding (#7155) and inline-blob attachments
  (#7300), CopilotStudio `LineTooLong` on large activities (#7417), mem0
  storage/search scope separation (#7531), Azure AI Search query-source
  identity (#7278).
- **MCP**: `header_provider` headers on the initialize handshake and ambient
  requests (#7305, #7218), tool-use sampling results (#7189).
- **New package**: `azure-cosmos-memory` context provider (#6719).
- **Verification pass**: upstream added a Mistral *chat* client (#7392); this
  port already has `MistralChatClient`, but it was written before upstream's and
  has not been diffed against it.

## Post-`68136ee` drift (checked against `beb65b21`, 2026-07-13)

Upstream moved 4 Python commits past the `68136ee` baseline; all four are
accounted for:

- **`as_tool` session propagation** (`f3057ef2`, fixing a feature already in
  `68136ee` that the port had not yet carried): `AsToolOptions` gained
  `propagate_session` (plus the previously missing `stream_callback` and
  `approval_mode`). Implemented with upstream's *fixed* child-session
  semantics: the sub-agent runs on an `AgentSession::child` of the parent —
  same `session_id`, **shared** `state` bag, **isolated** (cleared)
  `service_session_id`, so the parent's pending server-side conversation
  pointer never leaks into the sub-agent's own service calls. Plumbing:
  `AgentSession.state` became a `SessionState` handle (shared by reference
  across clones, matching Python's dict-reference semantics), the agent hands
  its session to the function-invocation loop via a non-wire
  `ChatOptions::session` side channel (popped before the provider client sees
  the options, exactly like upstream's client-kwargs `pop("session")`), and
  tools can read it through `FunctionInvocationContext::session` /
  `Tool::invoke_in_context`.
- **Parallel tool-span context** (`7f4cc296`): Python lost the ambient span
  when fanning parallel tool calls out via `asyncio.create_task` without
  copying contextvars. The Rust loop polls all invocations in-task under the
  instrumented future, so the parent span always propagates — no code change
  needed; a regression test now pins the behavior
  (`observability.rs::parallel_tool_call_spans_keep_the_surrounding_span_as_parent`).
- **Harness compaction fix** (`b3d523ee`): `@experimental` harness module —
  out of scope (see "Remaining").
- **OTel Distro sample** (`8e74360d`): Python-only sample — no Rust action.

A subsequent example-gallery audit against upstream's `python/samples` closed
two further gaps that predated the re-baseline:

- **Embeddings** (UPSTREAM_DRIFT §4/§5's "if in scope" item — now in scope):
  `Embedding`/`GeneratedEmbeddings`/`EmbeddingGenerationOptions` types + the
  `EmbeddingClient` trait in core, with provider clients for **OpenAI**
  (`/v1/embeddings`, loopback-tested), **Azure OpenAI** (deployment-scoped,
  api-key/Entra), **Ollama** (OpenAI-compatible surface), and **Mistral**
  (`mistral-embed` default — upstream's mistral package is embeddings-only).
  Bedrock/Foundry/Gemini embedding clients remain open (small, independent
  additions).
- **Progressive tool exposure** (upstream `FunctionInvocationContext.tools`):
  a `LiveToolList` handle on the invocation context with
  `add_tools`/`remove_tools` (duplicate-name rejection, batch-validated);
  the function-calling loop re-snapshots it at the top of every model
  iteration, so mutations take effect on the next iteration, never the
  in-flight batch.

## Done

### Naming / type-system cascade — Theme A + Theme F (complete)

`trait Agent`→`SupportsAgentRun`, `ChatAgent`→`Agent`, `ChatAgentBuilder`→
`AgentBuilder`, `ChatMessage`→`Message`, `AgentRunResponse`/`…Update`→
`AgentResponse`/`AgentResponseUpdate`, `ChatResponse.model_id`/`ChatOptions.model_id`→
`model`, `AiFunction`→`FunctionTool`, `AgentRunContext`→`AgentContext`,
`CitationAnnotation`→`Annotation`.

### Types & tools (§5/§6/§8)

- 12 new hosted tool-call/result `Content` variants; typed `UsageDetails`
  cache/reasoning fields (wired from Anthropic/OpenAI); `ContinuationToken`;
  `Annotation` `type:"citation"` discriminator.
- `hosted_image_generation()` + `ToolKind::HostedImageGeneration`.
- Cache/reasoning/`embeddings`/`prompt.name` OTel attributes.

### Sessions / context (§3) & new modules (§9)

- **ContextProvider → SessionContext reshape**: `Context`→`SessionContext`,
  `invoking`/`invoked`/`thread_created`→`before_run`/`after_run` (in-place
  mutation), `AggregateContextProvider` removed; ported across core + the
  redis/mem0/azure-ai-search provider crates.
- **`settings`** module (`SecretString` + `load_setting`).
- **`compaction`** module (`Tokenizer` + Truncation/SlidingWindow/TokenBudget/
  SelectiveToolResult strategies).

### Workflow engine & orchestrations (§10/§12)

- Per-executor serialization within a superstep; staged shared-state
  (commit-per-superstep).
- `WorkflowEvent::Intermediate` + `output_from`/`intermediate_output_from`
  designation + `OutputValidation`, wired through the Sequential/Concurrent/
  GroupChat/Magentic builders.
- Async edge conditions (`should_route`) with a backward-compatible sync API +
  `EdgeGroup::has_condition`.
- **Handoff mesh topology**: `add_handoff(src).to(targets)` edges are now
  enforced per-source (previously the adjacency map was built but discarded, so
  every agent could reach every other). A source is restricted to its declared
  outgoing edges; a source with no edges (when any edge is declared) is a leaf
  that cannot initiate a handoff; an empty map preserves the full-mesh
  back-compat. Rejected targets reuse the existing unknown-target feedback path.

### Providers & hosting (§13/§14)

- Removed the dead `OpenAIAssistantsClient`; flipped OpenAI client names
  (`OpenAIChatClient`=Responses, `OpenAIChatCompletionClient`=Chat Completions).
- New provider crates: **ollama**, **gemini**, **mistral**, **foundry-local**
  (Microsoft Foundry Local's OpenAI-compatible localhost endpoint; reuses
  `agent_framework_openai::convert`), **bedrock** (AWS Bedrock Converse
  API with a dependency-free **SigV4** signer verified against AWS's published
  `get-vanilla` known-answer test vector), and **github-copilot**
  (OpenAI-compatible chat endpoint behind the GitHub→Copilot short-lived-token
  exchange, with token caching/refresh) — full `ChatClient` impls, wired into
  the umbrella crate + examples.
- **`agent-framework-azure-ai` → `agent-framework-foundry`** (the largest
  provider item): upstream deleted the Azure AI Agents threads/runs data-plane
  and replaced it with the `foundry` package on the Responses API. Renamed the
  crate and rewrote it — `FoundryChatClient` (Responses API,
  `POST {endpoint}/openai/v1/responses`, Entra scope `https://ai.azure.com/.default`)
  delegates to the existing `agent_framework_azure::responses::AzureOpenAIResponsesClient`
  rather than reinventing the transport; added `PromptAgentDefinition`,
  `FoundryAgent` (a `SupportsAgentRun` realizing a Prompt Agent client-side) and
  `to_prompt_agent()`. Env prefix `AZURE_AI_`→`FOUNDRY_`,
  `model_deployment_name`→`model`. Rewired umbrella crate (feature/re-export),
  examples, and docs; the distinct `agent-framework-azure-ai-search` crate is
  untouched. (Binding to a server-hosted agent on the Foundry Agents
  control-plane is a documented extension point, not yet wired.)
- **Anthropic multi-cloud** (rework in place, no new crates): the `anthropic`
  crate is now a superset with `AnthropicBedrockClient` (AWS Bedrock
  `InvokeModel`, `anthropic_version: bedrock-2023-05-31`, reusing the verified
  `agent_framework_bedrock::sigv4` signer), `AnthropicVertexClient` (Vertex
  `:rawPredict`, `vertex-2023-10-16`, pluggable `VertexTokenProvider` for the
  Google OAuth token), and `AnthropicFoundryClient` (Entra via
  `agent_framework_azure::TokenCredential`; route/version overridable as a
  documented extension point). A shared `convert::build_cloud_request` omits the
  top-level `model` (it's URL-encoded) and stamps the per-cloud
  `anthropic_version`. Cloud-transport streaming is a documented single-update
  adaptation (the AWS event-stream / `:streamRawPredict` framing is a marked
  extension point). No dependency cycle (bedrock/azure don't depend back).
- `CosmosCheckpointStorage`; DevUI security middleware (Host-header
  anti-DNS-rebinding guard + optional bearer auth, opt-in).
- **Reusable Responses-conversion module** (`hosting::responses`): extracted the
  OpenAI-Responses wire types + conversion (`responses_to_run` /
  `responses_from_run`, `ResponsesRequest`, `ResponseObject`) out of the DevUI
  internals into a public, framework-agnostic module — mirroring upstream's
  `hosting-responses` package and resolving the crate's self-documented TODO.
  Pure refactor; DevUI `/v1/responses` wire output unchanged.

### Streaming API shape — Theme B (satisfied idiomatically)

Upstream's Python unifies buffered vs. streaming behind `run(stream=…)` /
`get_response(stream=…)`. Rust can't cleanly return either a buffered value or
a stream from one function keyed on a runtime bool, so the port already
expresses this idiomatically as method **pairs** — `run`/`run_stream` and
`ChatClient::get_response`/`get_streaming_response`. No further work: the
capability is present, just spelled the Rust way.

## Remaining

The tractable, verifiable alignment is complete. Everything still open falls
into one of three buckets — large-and-externally-blocked, or a deliberate,
documented divergence, or low-verifiability without an upstream artifact this
repo doesn't have. None is a straightforward port.

**Deliberate / documented divergences (not gaps to "fix"):**
- **Streaming API shape (Theme B)** — expressed as Rust method pairs
  (`run`/`run_stream`); a single `stream=`-keyed function isn't idiomatic Rust.
- **Declarative *workflow* DSL** — upstream's declarative workflow schema is the
  Power Platform / Copilot Studio imperative DSL, which doesn't map onto this
  port's graph engine; the crate defines a documented Rust-native `WorkflowSpec`
  instead. Agents and Rust-native workflows already load **and execute**.
- **Server-hosted control-plane bindings** left as documented extension points:
  the Foundry Agents control plane (`FoundryAgent` realizes a Prompt Agent
  client-side), and true incremental cloud-transport streaming (AWS
  event-stream framing / Vertex `:streamRawPredict`).

**Large, externally-blocked ecosystem packages (each a substantial new crate):**
- **`durabletask`** — durable agent/workflow hosting over Microsoft's Durable
  Task Framework via a gRPC sidecar (replay-safe orchestration + entity model).
  Blocked on the sidecar protocol/SDK; second-largest ecosystem item.
- **`@experimental` harness / security / evaluation** modules — upstream-unstable
  surfaces, low value to pin before they settle.
- **`agent-framework-claude`** — a `BaseAgent` that subprocesses the Claude
  Agent SDK / CLI; there is no Rust Claude Agent SDK, so this is a subprocess
  shim of speculative value (distinct from the `anthropic` chat client, which is
  done).

**Low-verifiability without the upstream frontend contract:**
- **DevUI's remaining ~17 UI routes** (conversations / deployments API) — a
  pre-existing gap that serves the bundled web UI; faithfully porting them needs
  the frontend's request/response contract, which isn't in this repo. The
  security-relevant middleware (Host-header guard + bearer auth) and the core
  entity/responses routes are already in place; the reusable Responses
  conversion (`hosting::responses`) is done.
