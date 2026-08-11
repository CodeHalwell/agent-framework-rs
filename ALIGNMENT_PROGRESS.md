# Alignment progress against current upstream

Tracks the re-baselining of `agent-framework-rs` onto current upstream, as
catalogued in [`UPSTREAM_DRIFT.md`](./UPSTREAM_DRIFT.md). Section numbers under
the `68136ee` heading refer to that document. Every item recorded as landed was
independently verified (full workspace build + `cargo test` + clippy
`--all-targets` + rustfmt, all green) before commit.

**Current upstream baseline: `266206e` (2026-08-07).** Sections are newest
first; each records the upstream revision it was checked against.

## Post-`4b1afd90` drift (checked against `266206e`, 2026-08-07)

Upstream moved **6 commits** after the `4b1afd90` baseline. All six are .NET;
no Python commit landed in this window. One was ported, five are not
applicable — and two of those five mark subsystems this port simply does not
have, which is recorded here as a gap rather than dressed up as parity.

### Ported this pass (1, with regression tests)

| Upstream | Change | Rust site |
|---|---|---|
| #7539 | **Usage aggregation across a looping run.** Upstream's invariant: when a component re-invokes an inner agent or chat client several times within one logical run, the usage it returns must cover the whole run. This port's function-invocation loop violated it — every exit path returned the final iteration's `ChatResponse` untouched, so a five-iteration tool run reported roughly a fifth of the tokens it spent, and the `gen_ai.usage.*` OTel metrics (which read `usage_details`) under-reported with it. Confirmed by probe before fixing: a two-call run reporting 100 then 200 input tokens surfaced 200. `accumulate_usage` now folds each iteration's usage into a running aggregate applied to whichever response the loop returns — the no-more-calls exit, the approval pause, the declaration-only hand-back, and the tools-disabled failsafe. `UsageDetails::add_assign` already carried the null-aware semantics upstream's `UsageAggregator.Combine` specifies, so `None` still means *not reported* rather than zero. | `core/client.rs` (`accumulate_usage`, the `get_response` tool loop) |

Verified: full workspace build, `cargo test --workspace --all-features`
(**1606 passing**, 3 of them new), `cargo clippy --all-targets --all-features`
clean, `cargo fmt --check` clean. All three new tests were confirmed to fail
against the pre-fix loop, so none of them is vacuous.

### Not applicable (5)

| Upstream | Why not |
|---|---|
| #7535 | **Declarative fenced-string parsing.** Replaces a backtracking regex in `TrimJsonDelimiter` with a linear scan, to deny a malformed fenced input the chance to trigger catastrophic backtracking. This port pulls in no regex engine at all (`regex` appears nowhere in the tree) and has no fenced-code-block trimmer — the declarative crate parses YAML and conditions directly. There is nothing here to harden. |
| #7567 | **`AgentIsolationKeyProvider` rename.** A .NET hosting rename across A2A task stores, AG-UI endpoints, and session stores. This port has no isolation-key concept in `agent-framework-hosting`. |
| #7525 | **Single source of conversation history for a hosted agent.** Rewires .NET's `AgentSessionStore` / `AgentFrameworkResponseHandler` for Foundry hosting. This port's hosting crate has no `AgentSessionStore` equivalent. |
| #7540 | **Hardened file skill discovery.** Skips symlinked `SKILL.md` files and symlinked subdirectories during discovery, and fails closed on paths it cannot inspect. Not applicable *because the port has no filesystem skill source at all* — `skills.rs` builds `Skill` values in memory from caller-supplied strings, so there is no discovery walk to harden. See the gap note below. |
| #7388 | **`InvocableFunctionBypassingChatClient`.** See the gap note below. |

### Gaps this pass surfaced (not closed)

Two upstream changes landed on subsystems this port lacks. Neither is a
regression, and neither was half-implemented to make the table look better:

- **File-based skills.** Upstream (both languages) discovers skills from disk:
  `SKILL.md` with YAML frontmatter, resource and script files found by
  scanning the skill directory, and a security boundary around path traversal
  and symlink escape. This port's `SkillsProvider` is in-memory only, so
  #7540 and the earlier #7507 (Windows junction detection) have no landing
  site. Adding a file source means adopting that whole security boundary, not
  just a directory walk.
- **Sibling backend calls dropped beside a declaration-only call.** When one
  response mixes an invocable tool call with a declaration-only (frontend)
  one, this port returns the whole response unexecuted — the same limitation
  #7388 works around in .NET. Upstream's fix is an opt-in decorator that
  stashes the invocable calls in the session state bag and re-injects them
  next turn as pre-approved approval responses. That mechanism depends on an
  `AgentSessionStateBag` and approval-response binding this port does not
  have, so it is a design task rather than a port, and is left open
  deliberately.

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
| #7095 | **Gemini 3 `thought_signature` replay.** Not supported at all. Gemini 3 requires the signature echoed when a function call is replayed, or the follow-up turn is rejected. Two placements exist and both are now handled: on the **function-call part itself** (Gemini 3's usual placement — `FunctionCallContent::protected_data`, which wins) and on a **preceding thought part** (`TextReasoningContent::protected_data`, used only to backfill a call carrying none, mirroring upstream's "backfill only when the raw Part lacks one"). Reasoning is no longer sent back as a part, matching upstream. | `core/types/content.rs`, `gemini/convert.rs` |

Verified: full workspace build, `cargo test --workspace --all-features`
(**1508 passing**, 24 of them new), `cargo clippy --all-targets --all-features`
clean, `cargo fmt --check` clean.

### Compaction cluster (follow-up pass)

Upstream's `_compaction.py` is an annotation-driven system that groups messages
into spans and flags them `_excluded`; this port deliberately implements a
smaller "return a reduced list" model (see the module docs). So the cluster was
triaged by *invariant* rather than by patch — what does upstream's fix
guarantee, and does this port's much simpler design guarantee it too? Two did
not, and both produced conversations providers reject outright:

| Upstream | Invariant | What was wrong here |
|---|---|---|
| #7406 | A function call and its result are retained or dropped **together**. | Confirmed broken two ways by probe. `TokenBudget` dropped an expensive call-bearing message while keeping its cheap result — a tool message answering nothing. `SelectiveToolResult` deleted stale results outright while their assistant `tool_calls` entries stayed — an unanswered call. Either half alone is a 400 on the next request, not merely a worse completion. Upstream enforces this by linking call and result into one indivisible span *before* any strategy runs; with no span model here, `drop_orphaned_tool_exchanges` enforces the identical observable guarantee as a repair pass over the retained set. |
| #7219 | Compaction never yields a projection with nothing for the model to answer. | `Truncation::new(1)` / `SlidingWindow::new(0)` over a conversation with a system prefix returned system messages *only*. `ensure_non_system_message` reinstates the most recent non-system turn, accepting a result over the limit exactly as upstream does. |

`SelectiveToolResult` changed shape as a result: it now **replaces** a stale
result's payload with `OMITTED_TOOL_RESULT` instead of deleting the content.
That keeps the exchange paired (no orphan to repair) while still shedding the
bulk, and matches the intent of upstream's `ToolResultCompactionStrategy`,
which likewise replaces stale tool groups with a compact stand-in rather than
removing them — upstream summarizes the group with an LLM; this port, having no
summarizing strategy, substitutes a fixed marker. Three existing tests asserted
the old delete-and-drop behavior over fixtures with unpaired results; they were
rebuilt on realistic paired conversations.

Ordering is load-bearing and documented at the call site: the orphan repair runs
*first*, because it can itself strip a conversation down to system-only (when
the sole non-system message was an orphaned result), which the minimum-retention
pass then catches.

Not applicable in this cluster:

- **#7124** (token counts inflated by `\uXXXX` escapes on non-ASCII text) —
  upstream counted tokens off a JSON serialization with `ensure_ascii=True`;
  this port counts message text directly, so the inflation never existed.
  Pinned with a regression test rather than changed.
- **#7391** (ignore `_excluded` tool results) — depends on upstream's
  exclusion-marking model, which this port does not implement.
- **#7396 / #7375** (bound tool-result summaries; bound summarization input
  before the provider call) — both govern the LLM-backed `Summarization`
  strategy, which this port does not have.

### mem0 storage/retrieval scope separation (#7531)

The port had the same cross-user memory leak upstream fixed: `before_run`
searched with `self.user_id` / `self.agent_id` — the *storage* scope. A provider
configured with a shared `agent_id` (one agent serving many users, the ordinary
deployment shape) therefore retrieved memories written by **every** user of that
agent and injected them into the current user's conversation.

`Mem0Provider` now separates the two scopes exactly as upstream does:

- **Storage** — `with_application_id` / `with_agent_id` / `with_user_id`, stamped
  onto memories written by `after_run`, never used to retrieve.
- **Retrieval** — `with_search_application_id` / `with_search_agent_id` /
  `with_search_user_id`, each queried as its **own** request and the results
  merged (Mem0 ANDs the entries of a single `filters` object, so one combined
  query would return only memories tagged with *both* — dropping exactly the
  agent-wide memories written by other users that `search_agent_id` exists for;
  upstream fans the partitions out for the same reason). Used only by
  `before_run`, and never inheriting from
  the storage scope. With no retrieval scope set, `before_run` retrieves nothing
  and warns once. `search_application_id` narrows a search only as a fallback
  when neither user nor agent retrieval scope is set, matching upstream.

This is a deliberate **behavior change**, as it was upstream: code that
retrieved memories via `with_user_id` alone must now also set
`with_search_user_id`. Agent-wide retrieval has to be requested explicitly,
which is the entire point — the leak came from it being implicit. Seven
loopback tests configured only a storage scope and were updated; three new tests
pin the isolation.

### Approvals: duplicate call on round-trip (#7271) — fixed

`replace_approval_contents_with_results` deduped a restored function call
against **only the message being scanned**. On an approval round trip a hosting
layer replays the stored `function_call` item and its approval request as two
*separate* assistant messages, so the per-message check never fired and the
approval request restored a second copy of the call. Only one copy received the
function result; the provider then rejects the orphan with "No tool output
found for function call ...".

Now collects pending call ids across all messages, excludes ids that already
carry a result (reusing a call id for a later invocation is supported, and a
completed pair must not suppress a fresh request), and records each restored id
so two approval requests for the same call cannot both expand. Four regression
tests cover the round-trip shape, the double-request case, the single-request
case that must still expand, and the reused-id case.

### Approvals: blocked on a missing core field (#7462)

Upstream stopped serializing **local** function approvals as MCP input items on
the OpenAI Responses path — local approvals are resolved in-process, and only
*hosted* (MCP) decisions have a matching approval request on the provider. It
distinguishes the two with `_is_hosted_tool_approval`, which tests
`function_call.additional_properties["server_label"]`.

This port cannot express that test: `FunctionCallContent` has only
`call_id` / `name` / `arguments` — no `additional_properties` — so hosted and
local approvals are indistinguishable, and `messages_to_input` serializes every
approval content as `mcp_approval_request` / `mcp_approval_response`.

In practice the local case is mostly shielded by layering, since
`FunctionInvokingChatClient` converts local approvals into calls and results
*before* the Responses client serializes anything, and an approved hosted
approval survives untouched (no local tool produces a result for it, so the
conversion is skipped). One narrower divergence is real and shares the same
blocker: a **rejected** approval is unconditionally converted into a local
rejection result, so a rejected *hosted* MCP approval never reaches the provider
as `mcp_approval_response {approve: false}`. Closing either needs
`additional_properties` on `FunctionCallContent` first — a core type change
worth doing deliberately rather than as a side effect.

### Provider cluster — triaged, six of seven not applicable

Worked through the provider fixes as a batch. Only one turned out to need code,
and it needed none: the rest do not apply to this port's architecture. Recorded
individually so the next pass does not re-derive them.

| Upstream | Verdict |
|---|---|
| #7199 — raw JSON-Schema `response_format` passed through unwrapped | **Structurally impossible here.** Upstream's bug needs a raw dict in the `response_format` slot; this port's `ResponseFormat` is a closed typed enum (`Text` / `JsonObject` / `JsonSchema{..}`) whose `Serialize` impl always builds the correct envelope. |
| #7163 — GPT-5.6 prompt-cache breakpoints | **Half already available, half blocked.** The request-wide `prompt_cache_options` reaches the wire today through `ChatOptions::additional_properties`, which `apply_options` merges into the body — no typed field needed, now pinned by a test. The *per-content* `prompt_cache_breakpoint` marker is blocked: it lives on `Content.additional_properties`, which this port's content types do not have. |
| #7283 — Foundry agent inheriting `OPENAI_CHAT_MODEL` | **N/A.** This port reads `FOUNDRY_MODEL` and never consults `OPENAI_CHAT_MODEL`, and the agent-*reference* request path the bug lives on is a documented unimplemented extension point here (`FoundryAgent` realizes a Prompt Agent client-side, where sending a model is correct). |
| #7417 — CopilotStudio `LineTooLong` on large activities | **N/A.** aiohttp's 512 KB per-line read buffer is the cause; this port speaks Direct-to-Engine over `reqwest`, which has no equivalent per-line cap. |
| #7155 — forward `GitHubCopilotOptions` verbatim to `create_session` | **N/A.** Different surface: this port's GitHub Copilot client is the OpenAI-compatible `POST /chat/completions` endpoint, not the Copilot Agent SDK's session API. |
| #7300 — forward Copilot input attachments as inline blobs | **N/A**, same reason as #7155 (`copilot_session.send(..., attachments=...)`). |
| #7278 — Azure AI Search query-source identity | **N/A for now.** The `x-ms-query-source-authorization` header applies to *agentic* Knowledge Base retrieval; this port's provider implements classic index search (hybrid/semantic/vector) only. Agentic retrieval is a feature gap, not a bug — worth its own decision. |

The recurring blocker is worth calling out on its own: **three separate items
now hinge on this port's content types lacking `additional_properties`** —
#7462 (hosted vs. local approvals), the rejected-hosted-approval divergence
found alongside it, and #7163's per-content cache breakpoints. Adding
`additional_properties` to `Content` / `FunctionCallContent` would unblock all
three at once and is the highest-leverage next piece of work in this area.

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

- **Approvals cluster** (#7407, #7408, #7410, #7345, #7090): decisions
  preserved under OpenAI continuation, tool content returned after invocation
  limits, provider-injected approvals deferred to in-run execution,
  resume/replay, auto-approval name-collision warnings. (#7271 is done — see
  above; #7462 is blocked — see below.)
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
- **Providers**: Responses conversation-ID helper (#7234, BREAKING). The rest
  of this cluster was triaged and is not applicable — see below.
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
