# Alignment progress against current upstream (`68136ee`)

Tracks the re-baselining of `agent-framework-rs` onto current upstream, as
catalogued in [`UPSTREAM_DRIFT.md`](./UPSTREAM_DRIFT.md). Section numbers below
refer to that document. Everything under "Done" is landed on
`claude/rust-agent-framework-alignment-q6bjwp` and independently verified
(full workspace build + `cargo test` + clippy `--all-targets` + rustfmt, all
green) before commit.

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

### Providers & hosting (§13/§14)

- Removed the dead `OpenAIAssistantsClient`; flipped OpenAI client names
  (`OpenAIChatClient`=Responses, `OpenAIChatCompletionClient`=Chat Completions).
- New provider crates: **ollama**, **gemini**, **mistral** (full `ChatClient`
  impls, wired into the umbrella crate + examples).
- `CosmosCheckpointStorage`; DevUI security middleware (Host-header
  anti-DNS-rebinding guard + optional bearer auth, opt-in).

## Remaining (roadmap)

Larger, higher-risk or provider-API-specific efforts, roughly by leverage:

1. **Sessions §2 completion** — `AgentThread`→`AgentSession` (state bag, sync
   `to_dict`/`from_dict`) and history out of the thread into a
   `HistoryProvider` (`InMemory`/`File`), rewiring the agent run loop. Broad
   ripple (a2a/copilotstudio/hosting/redis/examples).
2. **Unified `run(stream=)` / `get_response(stream=)`** (Theme B) — cross-cutting.
3. **Provider reworks** — `azure-ai`→`foundry` rename+rewrite onto the Responses
   API (`FoundryAgent`/`to_prompt_agent`); Anthropic multi-cloud
   (Bedrock/Vertex/Foundry); Azure "routing mode" realignment; remaining new
   provider crates (bedrock, foundry-local, claude, github-copilot).
4. **Orchestration depth (§12)** — shared post-agent `AgentApprovalExecutor`
   HITL (iterate-until-approved); terminal output shape (`AgentResponse` vs
   transcript); Handoff mesh-topology rebuild.
5. **Hosting/DevUI (§14)** — remaining ~17 DevUI routes; Responses-conversion
   extraction; A2A server move.
6. **Large ecosystem packages** — `durabletask`; the declarative-workflow
   execution engine; the `@experimental` harness / security / evaluation modules.
