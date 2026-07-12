# Alignment progress against current upstream (`68136ee`)

Tracks the re-baselining of `agent-framework-rs` onto current upstream, as
catalogued in [`UPSTREAM_DRIFT.md`](./UPSTREAM_DRIFT.md). Section numbers below
refer to that document.

## Done (landed on `claude/rust-agent-framework-alignment-q6bjwp`, all green)

Every item here builds clean and passes the full workspace suite (67 test
binaries), clippy `--all-targets`, and rustfmt.

### Naming / type-system cascade — Theme A + Theme F (complete)

| Change | Section |
|---|---|
| `trait Agent` → `SupportsAgentRun` | §1 / Theme A |
| `ChatAgent` → `Agent`, `ChatAgentBuilder` → `AgentBuilder` | §1 |
| `ChatMessage` → `Message` | §5 |
| `AgentRunResponse`/`…Update` → `AgentResponse`/`AgentResponseUpdate` | §5 |
| `ChatResponse.model_id` / `ChatOptions.model_id` → `model` | §5 |
| `AiFunction` → `FunctionTool` | §6 |
| `AgentRunContext` → `AgentContext` | §7 |
| `CitationAnnotation` → `Annotation` (+ `type:"citation"` discriminator) | §5 |

### Type-system additions (§5)

- Twelve new hosted tool-call/result `Content` variants
  (code-interpreter, image-generation, MCP, search, shell ×2, shell-command-output,
  oauth-consent) with exact upstream wire tags — previously dropped by the
  `Unknown` fallback.
- `UsageDetails` typed `cache_creation` / `cache_read` / `reasoning` token
  fields, populated by the Anthropic and OpenAI usage parsers.
- `ContinuationToken` newtype + optional field on `ChatResponse` / `AgentResponse`.

### Correctness fixes (§10 / §11)

- Per-executor serialization within a superstep (upstream PR #6776).
- Staged shared-state with commit-per-superstep (Pregel `_state.py` model).

### Providers (§13)

- Removed the dead `OpenAIAssistantsClient` (upstream deleted the Assistants API).
- Flipped OpenAI client names: `OpenAIChatClient` = Responses,
  `OpenAIChatCompletionClient` = Chat Completions.

### New capabilities

- `hosted_image_generation()` tool + `ToolKind::HostedImageGeneration` (§6).
- `settings` module: `SecretString` + `load_setting` precedence (§9).
- Cache/reasoning/`embeddings`/`prompt.name` OTel attributes (§8).

## Remaining (large, dedicated efforts — ordered by leverage)

Each is a multi-file change; several ripple into test-heavy crates and are best
done as focused, reviewed passes rather than a single sweep.

1. **Sessions keystone (§2 / §3)** — `AgentThread` → `AgentSession` (history
   leaves the thread), `ChatMessageStore` → `HistoryProvider`
   (`InMemory`/`File`), `Context` → `SessionContext`, `invoking`/`invoked` →
   `before_run`/`after_run` (in-place mutation), remove `thread_created` and
   `AggregateContextProvider`. Rewrites the agent run loop and the
   `redis` / `mem0` / `azure-ai-search` `ContextProvider` impls (+ their test
   suites). Unblocks compaction, skills, security, harness, and the
   history-provider ecosystem reworks.
2. **Unified `run(stream=)` / `get_response(stream=)`** (Theme B) — folds the
   paired streaming entry points into one overloaded call across every client,
   agent, and workflow surface.
3. **History compaction (§9)** — new `compaction` module (`TokenizerProtocol`,
   `CompactionStrategy`, seven strategies, `CompactionProvider`), wired into the
   client loop. Gated on the sessions keystone.
4. **Provider reworks (§13)** — rename+rewrite `azure-ai` → `foundry`
   (`FoundryChatClient` / `FoundryAgent` / `to_prompt_agent`), Anthropic
   multi-cloud (Bedrock/Vertex/Foundry), Azure "routing mode" realignment, and
   the seven new provider crates (`claude`, `bedrock`, `gemini`, `mistral`,
   `ollama`, `github-copilot`, `foundry-local`).
5. **Workflow / orchestration (§10 / §12)** — `output_from` /
   `intermediate_output_from` designation + `WorkflowEvent::Intermediate`, async
   `should_route`, `OUTPUT_VALIDATION`, the shared post-agent
   `AgentApprovalExecutor` HITL engine, and the Handoff mesh-topology rebuild.
6. **Hosting / DevUI (§14)** — Responses-conversion extraction, A2A server
   move, DevUI auth-by-default + anti-DNS-rebinding Host-header middleware, and
   the remaining ~17 DevUI routes.
7. **New ecosystem packages (§9 / §14)** — `durabletask`, `azure-cosmos`
   (`CosmosCheckpointStorage` is unblocked today), the declarative-workflow
   execution engine, shell-tools crate, and the `@experimental` harness /
   security / evaluation modules.
