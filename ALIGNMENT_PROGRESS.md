# Alignment progress against current upstream (`68136ee`)

Tracks the re-baselining of `agent-framework-rs` onto current upstream, as
catalogued in [`UPSTREAM_DRIFT.md`](./UPSTREAM_DRIFT.md). Section numbers below
refer to that document. Everything under "Done" is landed on
`claude/rust-agent-framework-alignment-q6bjwp` and independently verified
(full workspace build + `cargo test` + clippy `--all-targets` + rustfmt, all
green) before commit.

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
