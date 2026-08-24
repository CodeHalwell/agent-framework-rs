# Alignment progress against current upstream

Tracks the re-baselining of `agent-framework-rs` onto current upstream, as
catalogued in [`UPSTREAM_DRIFT.md`](./UPSTREAM_DRIFT.md). Section numbers under
the `68136ee` heading refer to that document. Every item recorded as landed was
independently verified (full workspace build + `cargo test` + clippy
`--all-targets` + rustfmt, all green) before commit.

**Current upstream baseline: `a63d462` (2026-08-24).** Sections are newest
first; each records the upstream revision it was checked against.

## Post-`e1326eb` drift (checked against `a63d462`, 2026-08-24)

The `e1326eb` pass below was triaged against a fork mirror that had stopped
advancing on 2026-08-20; the sync then caught up and moved **31 further
non-merge commits** (through 2026-08-24). One lands on this port.

### Ported this pass (1, with regression tests)

| Upstream | Change | Rust site |
|---|---|---|
| #7673 | **The GenAI semantic-convention version was never selectable, and the provider tag was emitted twice.** `gen_ai.system` was renamed to `gen_ai.provider.name` above the OTel v1.36.0 baseline. This port emitted **both** names on every chat span (and `gen_ai.provider.name` on the metrics attributes), so a consumer pinned to the baseline saw an attribute its version does not define, and one on the latest saw a name that had been renamed away. Upstream's fix makes the version an explicit input: `OTEL_SEMCONV_STABILITY_OPT_IN`, a comma-separated opt-in list in OpenTelemetry's standard format, whose `gen_ai_latest_experimental` token selects the conventions above the baseline — defaulting, when unset, to *opted in*, which upstream documents as a deliberate departure from OpenTelemetry's own default. `ObservabilityConfig` now carries that value and derives `use_latest_experimental_gen_ai_semconv()` / `emit_tool_call_attributes()`; exactly one provider attribute is emitted, and the four above-baseline attributes (`cache_creation.input_tokens`, `cache_read.input_tokens`, `reasoning.output_tokens`, `tool.definitions`) plus `gen_ai.tool.call.arguments`/`result` are withheld at the baseline. Under the default the only visible change is that `gen_ai.system` no longer rides along beside `gen_ai.provider.name`. | `core/observability.rs` (`GEN_AI_LATEST_EXPERIMENTAL_OPT_IN`, `ObservabilityConfig`, `chat_span`, `record_request`, `record_response`, `record_tool_arguments`, `record_tool_result`, `ObservableChatClient`, `metrics::record_chat_completion`), `core/client.rs` (tool-span call sites) |

The recording functions took a `capture_content: bool`; they now take
`&ObservabilityConfig`, because the second gate is not a property of the call
site. That keeps both gates explicit and, unlike reading the environment inside
the recorders, leaves them free of hidden global state on a per-response path.
`ObservableChatClient::with_content_capture` still works and sets the flag on
the config; `with_observability_config` sets both.

Verified: full workspace build, `cargo test --workspace --all-features`
(**1645 passing**, 4 of them new), `cargo clippy --all-targets --all-features`
under `-D warnings` (CI's own flag) clean, `cargo fmt --check` clean. All four
new tests were confirmed to fail against ungated code.

### Not applicable (30)

| Upstream | Why not |
|---|---|
| #7799, #7801 | **MCP tool argument shadowing the remote tool name**, and its documentation follow-up. Python's generated MCP function held the remote tool name as a keyword-only parameter *default*, and model-supplied arguments are splatted into that function — so an argument named `_remote_tool_name` bound to the parameter and redirected the call to a different remote tool. This port's MCP tools capture the remote name in the tool struct and pass arguments as one `serde_json::Value` map to `call_tool(name, arguments)`; there is no splat and no parameter for an argument to bind to. |
| #7289 | **Turn-scoped `after_run` providers deferred to the agent-loop boundary.** Each `AgentLoopMiddleware` iteration is a full agent run, so `CompactionProvider.after_run` fired per iteration and rewrote persisted history mid-task; providers can now opt into once-per-turn semantics. The port has no harness agent loop (a documented "remaining" item), so nothing drives several runs inside one turn and there is no per-iteration re-fire to defer. |
| #7625 | **GitHub Copilot telemetry config forwarding**, plus the Python settings-machinery fixes it needed (parameterized generics and `Literal` arms in runtime annotation checks). Both halves are Python-shaped: this port's `agent-framework-github-copilot` targets the OpenAI-compatible chat endpoint rather than the Copilot Agent SDK's session API — the same reason #7155 and #7300 were not applicable — and its settings are typed struct fields, not runtime-inspected annotations. |
| #7779 | **DevUI forwards `function_invocation_kwargs` to `agent.run`.** No `function_invocation_kwargs` concept here: tools receive a typed `Value` plus a `FunctionInvocationContext`. |
| #7734 | **FoundryEvals always emits an `arguments` field for tool calls**, in `_evaluation.py`. No evaluation crate. |
| #7423 | **A2UI (Agent-to-UI) support in the AG-UI adapter.** No AG-UI crate. |
| #7670, #7370, #7649 | **Foundry hosted-agent resiliency, steerable hosted agents, hosted state persistence** (Python and .NET). This port has no Foundry hosted-agents host. |
| #7768 | **Pin GitHub Actions to full-length commit SHAs**, across upstream's own 24 workflow files. This is repo hygiene rather than framework parity; worth adopting for this repo's two workflows, but resolving each action's SHA means reading repositories outside this session's GitHub scope, so it is left as a follow-up rather than guessed at. |
| #7812, #7814, #7795, #7813, #7804, #7754, #7678 | Release version bumps (Python 1.15.0, .NET 1.19.0), release-tag resolution, a DevFlow command fix, codeowners, README/doc edits. |
| #7774, #6441, #1893, #7709, #7639, #7778, #7829 | .NET: the MCP long-running-task migration to the 2026-07-28 Tasks extension, GitHub Copilot `ReasoningSummary` passthrough, Azure Blob Storage session persistence, a feature-usage bitmask, and dependency bumps. |
| #7780, #7781, #7782, #7783, #7784 | Python tooling bumps (`uv`, `ruff`, `ty`, `mypy`, `flit-core`). |

## Post-`5c06755` drift (checked against `e1326eb`, 2026-08-20)

Upstream moved **38 non-merge commits** in this window (2026-08-16 → 08-20).
Two land on this port: a bug it shares with upstream, and a capability its
middleware contract lacks. The other 36 are .NET, AG-UI, samples, dependency
bumps, or Python-shaped problems that cannot arise here — several of those
because the port's payloads are untyped `serde_json::Value`s rather than
dynamically resolved Python types.

### Ported this pass (1 fix + 1 capability, both with regression tests)

| Upstream | Change | Rust site |
|---|---|---|
| #7242 | **Replaying a conversation duplicated stored history.** A history provider is handed a run's input messages plus its response messages, so a caller that keeps its own transcript and replays all of it every turn (the AG-UI shape, and any client tracking history itself) hands back everything the provider already stored. Every provider appended it unconditionally, so history grew superlinearly — and because `before_run` prepends stored history to the request, the duplicated turns were resent to the model on every later run. `filter_new_messages` locates the stored run inside the incoming one and returns only what follows it. Matching is by `message_id` where a message has one and by role + contents where it does not, mirroring upstream's `get_message_identity`; the scan is not anchored at offset 0, so a provider whose stored history is a *trimmed window* (a retention limit having dropped the oldest messages) still aligns. Applied to all four history providers, not just the two upstream touched: `RedisChatMessageStore` and `CosmosChatMessageStore` have the identical shape and the identical bug, and now read their stored history before writing — one extra round trip per run, the same read `before_run` already makes. | `core/history.rs` (`filter_new_messages`, both providers' `after_run`), `redis/chat_message_store.rs`, `cosmos/chat_message_store.rs` |
| #7562 | **Function middleware had no way to fail closed.** The invocation loop converts every error a tool or its middleware produces into a `FunctionResultContent { exception, .. }`, hands it to the model and keeps looping. That is right for a tool failure the model can recover from, but an enforcement layer — a guardrail, a policy or authorization gate — needs the opposite: when it refuses a call, the run must stop, not hand the model an error string it can retry around. `Error::MiddlewareFailure` is the escape, and the only error the loop propagates rather than absorbs. Because the parallel batch runs under `try_join_all`, propagating it also drops the siblings still in flight — upstream's "cancel the in-flight batch" without needing a cancellation mechanism of its own. | `core/error.rs` (`MiddlewareFailure`, `middleware_failure`, `is_middleware_failure`), `core/client.rs` (`execute_tool_call`), `core/observability.rs` (`error_type`) |

Two review findings on PR #16 extended the fix past what upstream's own
patch covers, both confirmed by probe before fixing:

- **Alignment must not see response messages.** Concatenating a run's input and
  its responses before aligning let a response that coincidentally reproduced
  the stored tail match as a replay, swallowing the genuinely new input in
  front of it — stored `[q, a]` plus a run whose input is `q` and whose
  response opens with `a` stored nothing but the tail. Responses are generated
  by the run reporting them and can never be a replay, so `new_run_messages`
  aligns the input alone and appends every response.
- **The request side duplicated too.** Storing only the new suffix left the
  other half of the problem untouched: the agent sends injected context
  followed by the caller's input, so a provider that injected unconditionally
  sent `q1, a1, q1, a1, q2` for a caller replaying `q1, a1, q2` — verified end
  to end against the messages the model actually received. All four providers
  now inject nothing when the input already aligns against what they hold; the
  first fix covered only the two core ones, and a third review round caught
  that the Redis and Cosmos stores still had the unconditional injection.

A second review round raised two more, both fixed:

- **Which occurrence of a stored run to align on depends on what the provider
  holds.** A forward scan takes the first match, which is right for a store
  that keeps everything — a later match would discard the genuinely new turns
  in between — but wrong for a retention-limited one, whose stored list is a
  *window* of the most recent messages: there the last match is the window, and
  taking an earlier one re-pushes the whole middle of a replayed transcript on
  every turn. `StoredHistory::{Complete, Window}` makes it the caller's
  decision, and the Redis store picks `Window` only when its list is *at* its
  cap, since only a trimmed list can be a window.
- **A configured semconv version has to reach tool spans.** The tool loop
  rebuilt an `ObservabilityConfig` from the environment per call, so a client
  configured for one convention version could emit chat spans under it and tool
  spans under another. `FunctionInvokingChatClient` now carries the config
  (`with_observability_config`), resolved once at construction, and
  `AgentBuilder::observability_config` reaches that wrapper — the builder
  constructs it internally, so without a way through it the setter was
  unreachable on the main path.
- **The injection fix had to reach the remote stores.** It landed in the two
  core providers and stopped there, and the end-to-end test covered only the
  in-memory one, so a replay through a Redis- or Cosmos-backed session was
  still sent to the model twice. `inject_stored_history` is now public and used
  by all four, with a `StoredHistory`-aware variant for the Redis store.

The remote stores also no longer read before writing when there is nothing to
write (an empty run, or a Redis store configured to retain nothing, which is
documented to leave Redis untouched); both cases are pinned by pointing a store
at an address nothing is listening on and asserting the run still succeeds.
Their read-then-write sequence is not atomic, and the Cosmos read can lag a
just-landed write on an account with session or eventual consistency; both are
documented at the call sites rather than closed, since the fallback in each
case is the duplicate that the blind append they replace produced every time.

Two deliberate divergences in the dedup, both refusing to drop a turn that
might be real. Upstream falls back, when alignment fails, to deduplicating by
identity against a set of everything stored; that collapses two identical,
id-less `"yes"` turns into one and loses the second permanently. And upstream
treats an alignment consuming *all* of the incoming run — a run whose messages
exactly repeat the stored tail — as a replay carrying nothing new, storing
nothing; since a provider only reaches `after_run` by completing a real run,
this port reads it as a turn that genuinely repeated itself and stores it. In
both cases the port's behavior is what it was before the fix (append), so
neither can regress a conversation that used to be stored correctly.

The fail-closed signal is carried by the error *type* rather than by who
produced it, which is the one place it is looser than upstream's exception
class: a tool executor returning `Error::MiddlewareFailure` propagates the same
way. That is documented on the variant rather than guarded against — the
alternative (a marker only the pipeline can set) would need a wrapper type
threaded through every middleware signature for no practical gain.

Verified: full workspace build, `cargo test --workspace --all-features`
(**1656 passing**, 25 of them new), `cargo clippy --all-targets --all-features`
clean, `cargo fmt --check` clean. The two Redis tests run against a real
`redis-server` spawned by the existing integration harness; the Cosmos test
asserts the write count on the loopback server, so it fails if a replayed
message is written a second time. Both fixes were probed against the code they
fix: 5 of the history tests and both fail-closed tests fail without them (the
fail-closed pair by hanging on the 30-second sibling call, which is the
cancellation the fix buys). The remaining tests are negative controls — an
append-only run still accumulates every turn, an unalignable run is still
stored whole, and an ordinary middleware error is still absorbed into a
tool-error result and the loop still continues.

### Not applicable (36)

Grouped by why, rather than one row each:

| Upstream | Why not |
|---|---|
| #7684, #7500, #7636 | **Python type resolution.** Coercing JSON workflow-resume payloads into declared annotations, restricting request-info type-name resolution to caller-provided mappings, and a global checkpoint type registry. All three exist because Python resolves a payload's type *by name* at runtime. This port's `PendingRequest.request_data` and `RequestResponse.data` are `serde_json::Value`; there is no declared response type to coerce to, no type name in the payload to resolve, and no import to restrict — the executor deserializes what it asked for. |
| #7730 | **Structured instructions coerced to their `repr` when merged.** Upstream's `instructions` is declared `str` but widened by some clients to provider-native structured blocks, and three merge paths joined it with an f-string. `ChatOptions::instructions` is `Option<String>` here, so there is no non-string value to stringify; the newline concatenation in `ChatOptions::merge` and `prepare_request` is correct for every value the type admits. |
| #7755 | **`HandoffBuilder` clones dropped `Agent.additional_properties`.** Upstream's `HandoffAgentExecutor` rebuilds each participant agent to attach handoff tools. This port's `HandoffBuilder` holds `Arc<dyn SupportsAgentRun>` participants and never rebuilds them, and `Agent` carries no `additional_properties` field to lose. |
| #7557 | **Fan-in dropped all but the first trace context.** Upstream's workflow messages carry `trace_contexts` / `source_span_ids` lists for distributed-trace linking across a fan-in. This port's workflow engine propagates no trace context on messages at all — a standing gap in workflow observability, not a bug in aggregation, and one this commit does not close. |
| #7761 | **A2A input handling in orchestrations.** Three-part change: reject an empty A2A invocation explicitly, translate a remote `INPUT_REQUIRED` task into the `user_input_request` content contract so a group chat pauses on it, and restore that pending input from a checkpoint. The first half is already satisfied: `A2AAgent::run` errors on an empty `messages` list rather than inventing input (upstream was raising a bare `ValueError` and now raises `AgentInvalidRequestException` with session context; this port's message is already specific). The rest is blocked — the port's `A2AAgent` surfaces an `INPUT_REQUIRED` task's status message as ordinary chat messages, and there is no `user_input_request` content classification to translate the task into, so pausing an orchestration on remote input is a design task (tracked below), not a port. The two core-workflow hunks in this commit ride on the same classification. |
| #7766, #7510, #7662 | **AG-UI.** Unchanged predictive-state snapshots, tool-message IDs across snapshots, run continuity. No AG-UI crate. |
| #7606 | **A2A preview consent URLs**, in `foundry_hosting`. This port has no Foundry hosted-agents host. |
| #7698, #7695, #7693, #7706, #7746, #7740, #7762 | Harness blog samples, skill-script argument guidance, docs link fixes, spec/review-process guidance, engineering-system metadata, codeowners. |
| #7722, #7741, #7764, #7742, #7564, #7668, #7295, #7737, #7731, #7641, #7721, #7713, #7674, #7412, #7648, #7650 | .NET: A2A streaming artifacts, AG-UI history and SDK bumps, agent-hooks interception (the .NET half of #7515, already tracked as open), Foundry hosted samples and identity pass-through, `IServiceProvider` overloads, harness tool descriptions, session-persisted routing, release/build/version chores, declarative samples, Cosmos chat-history retrieval, and opt-in concurrent tool invocation (this port's loop is concurrent by default). |
| #7644, #7645 | Dependency bumps confined to Python tooling (`ty`, `flit`). |

### Standing gaps, reconfirmed (not closed)

- **Workflow trace propagation.** #7557 is the first upstream commit in this
  window to touch machinery — per-message `trace_contexts` carried across
  edges and merged at a fan-in — that the port's workflow engine does not have
  at all. Agent and tool spans are instrumented; workflow message flow is not.
- **`user_input_request` content classification.** #7761 needs an A2A
  `INPUT_REQUIRED` task to become a content item an orchestration recognizes
  as a request for caller input. The port has request-info events but no
  content-level classification for them, so an A2A participant cannot pause a
  group chat for remote input. Left open rather than half-built.

## Post-`2eb8fbb` drift (checked against `5c06755`, 2026-08-16)

Upstream moved **29 non-merge commits** in this window (2026-08-10 → 08-16),
the largest batch since the daily sync was repaired. Two are real bugs this
port shared and has now fixed; one is a mapping the port already had, now
pinned by a test; the rest are not applicable, and three of those land on
subsystems already tracked as open gaps.

### Ported this pass (2 fixes + 1 pinned mapping, all with regression tests)

| Upstream | Change | Rust site |
|---|---|---|
| #7470 | **A Redis retention limit of zero retained everything.** The documented sentinel for *unlimited* is not calling `with_max_messages` at all, so `0` must retain nothing — it retained every message instead. Trimming to `-(max)` emits `LTRIM key 0 -1` for `max == 0`, which is Redis's "keep the whole list", while the `len > max` guard is true for any non-empty list: the trim ran on every save and did nothing. `add_messages` now short-circuits *before* serializing, so no payload reaches Redis — or an AOF or replica — even briefly. It deliberately does **not** delete the key: `redis_key` is `{key_prefix}:{session_id}` with no per-provider discriminator, so two stores sharing a prefix and session id address the same list, and deleting would drop a co-located store's just-written history. Removing stored history is what `clear` is for. Upstream's other half — rejecting a *negative* limit, which emitted `LTRIM key 5 -1` and deleted the five oldest messages on every save — cannot arise here: `max_messages` is a `usize`. | `redis/chat_message_store.rs` (`add_messages`, `with_max_messages`) |
| #7546 | **Gemini 3 thought signatures were dropped across an approval round trip.** Gemini 3 rejects a request whose `functionCall` parts lack the `thoughtSignature` they were issued with. This port paired a signature to its call by *adjacency* only — a reasoning carrier immediately preceding the call — and cleared the held signature on any intervening content. Two failures followed, both ending in a 400 on the next turn: content that emits no Part at all (a `FunctionApprovalResponse`) cleared the signature merely by sitting between the carrier and its call; and a call replayed in a later message, with no carrier beside it, could never be signed. The clear now happens only for content that actually reaches the wire, and a `call_id -> signature` map accumulated across the conversation is the final fallback, which is what lets a later replay find its signature. Precedence is unchanged and matches upstream: the call's own `protected_data` wins, then an adjacent carrier, then the map. | `gemini/convert.rs` (`message_contents_to_parts`, `messages_to_gemini`) |
| #7597 | **Mistral prompt-cache usage** — already mapped, now asserted. Upstream's Mistral package hand-rolls its own usage mapping and had to grow `prompt_tokens_details.cached_tokens`; this port's `parse_response` delegates to the OpenAI parser, whose usage handling already covers it, so the field was never dropped. The delegation is the only reason there is no bug, so it is now a tested contract rather than an inherited accident. Upstream's explicit `isinstance(..., int) and not isinstance(..., bool)` guard has no counterpart here — `Value::as_u64` rejects strings, floats, and bools structurally — which is also pinned. | `mistral/convert.rs` (tests only; no behavior change) |

The Gemini fix diverges from upstream deliberately. Upstream caches signatures
in a bounded (256-entry, LRU) map on the *client*, which spans conversations
and therefore needs eviction and a `max_tracked_thought_signatures` knob.
Scoping the map to the conversation being converted is naturally bounded by the
history resent on every stateless request, needs no eviction policy, and cannot
leak a signature between conversations. The tradeoff, recorded rather than
hidden: a signature whose carrier has been dropped from history entirely is not
recoverable here, where upstream's client-lifetime cache would still hold it.

The map is written by the emit walk itself rather than gathered by a pre-pass.
The first cut used a pre-pass shaped like `collect_call_names`, and PR #15
review caught the flaw: a pre-pass has to *restate* the pairing rules, and that
restatement omitted the clear on wire-visible content. Because the map is
consulted only after adjacency has deliberately declined to sign a call, the
laxer map silently overrode that decision and re-signed calls the converter had
just refused — `[reasoning(sig), text, call]` reached the wire signed. Since a
replayed call always follows the turn that issued it, one forward pass suffices,
so the rules now have exactly one implementation and cannot drift apart.

Verified: full workspace build, `cargo test --workspace --all-features`
(**1627 passing**, 10 of them new), `cargo clippy --all-targets --all-features`
clean, `cargo fmt --check` clean. The two Redis tests run against a real
`redis-server` spawned by the existing integration harness. All six
bug-fixing tests were confirmed to fail against the code they fix — including
the two covering the review finding, checked against a reinstated lax map; the
remaining four are negative controls (an unsigned call stays unsigned, one
carrier never signs a second call, non-integer cached tokens are ignored) and
pass either way by design.

### Not applicable (26)

Grouped by why, rather than one row each:

| Upstream | Why not |
|---|---|
| #7486, #7608, #7533-adjacent FHA work | **Foundry hosted-agents host.** `_OutputItemTracker` duplicate-call suppression, FHA session-id translation. This port has no `foundry_hosting` equivalent — `agent-framework-foundry` is the client only. |
| #7655, #7594, #6646 | **AG-UI.** URL-source attachments, approval lifecycle/resume hardening, workflow checkpointing in the AG-UI adapter. No AG-UI crate. |
| #7652 | **DevUI frontend** streamed-tool-call dedup, in the bundled web UI this port does not ship. |
| #7622 | **MCP archive rejection warnings.** Raises `debug` to `warning` in `_ArchiveEntryLoader`, part of the file-based skill discovery subsystem this port lacks entirely (see the standing gap below). |
| #7631, #7607 | **Approval storage and approve-for-session scoping.** Both build on an `AgentSessionStateBag`-backed permission store this port has no equivalent of; related to the standing declaration-only gap. |
| #7521 | **[BREAKING] Require building functional workflow instances.** Python's `@executor`-decorated functions must now be built into instances before use — a Python-decorator ergonomics constraint. This port's `WorkflowBuilder` already requires constructed executors; there is no unbuilt form to reject. |
| #7550 | **JSON parsing for declarative workflows** — the Power-Platform-style declarative *workflow* DSL, already tracked as a deliberate divergence. |
| #7635 | **Cosmos memory provider** calling a renamed `add_cosmos` toolkit API — a Python-package rename with no Rust counterpart. |
| #7404 | **ClaudeAgent SDK client reuse** — the `agent-framework-claude` subprocess shim, a standing roadmap item. |
| #7450, #7602 | **BackgroundAgentsProvider `release_session`** (Python and .NET) — no background-agents provider in this port. |
| #7509, #7558, #7621, #7661, #7660, #7646, #7666, #7572, #7612, #7609, #7552 | Workspace glob matching, feature-usage telemetry, agentserver/package version bumps, code-owner enforcement, nuget config, .NET sample style, and .NET-only hosting/telemetry/diagnostics changes. |
| #7529, #7493, #7541, #7554, #7545 | Dependency bumps confined to Python tooling and the DevUI frontend (postcss ×2, pyrefly, js-yaml, zuban). |

### Standing gaps, reconfirmed (not closed)

This window added evidence to two gaps already on the books, and neither was
half-implemented to improve the table:

- **File-based skills.** #7622 is the third upstream commit in a row
  (after #7540 and #7507) to harden a discovery walk this port does not have:
  `SkillsProvider` builds `Skill` values in memory from caller-supplied
  strings. Adding a file source means adopting the whole security boundary —
  path traversal, symlink escape, archive size and format limits — not just a
  directory walk.
- **Declaration-only sibling calls.** #7631 and #7607 both extend the session
  state bag and approval-response binding that upstream's #7388 workaround
  depends on. That machinery is still absent here, so the gap widened rather
  than closed.

## Post-`266206e` drift (checked against `2eb8fbb`, 2026-08-11)

Upstream moved **8 commits** in this window — the first batch to arrive after
the fork's daily sync was repaired (it had failed on every run since it was
added, so the three preceding passes were all triaged against a mirror that
had stopped advancing on 2026-08-06).

**Nothing in the batch needed porting.** Five are dependency bumps confined to
Python tooling and the DevUI frontend (postcss ×2, pyrefly, js-yaml, zuban).
The three substantive commits are all Python and all land on subsystems this
port does not have:

| Upstream | Why not applicable |
|---|---|
| #7550 | **JSON parsing for declarative workflows.** Rewrites how an agent's free-text output is coerced to JSON — fenced-block extraction, then a scan for the last decodable object in prose — inside `_executors_agents.py`. That file belongs to the Power-Platform-style declarative *workflow* DSL, where an action captures an agent's output into a typed variable. This port's declarative crate is a spec model that compiles YAML into a `WorkflowBuilder` graph and passes conversation messages between agent nodes; it never JSON-decodes agent prose. Already tracked as an open roadmap item ("the upstream Copilot-Studio declarative *workflow* DSL"). |
| #7533 | **FHA migrated to `responses==2.0.0b1`, plus a Foundry state store.** Confined to the `foundry_hosting` package. This port has no Foundry hosted-agents host — `agent-framework-foundry` is the persistent-agents *client* only. |
| #7536 | **Encrypted reasoning made opt-in for Foundry chat.** Strips `reasoning.encrypted_content` from the `include` that upstream's base Responses client adds implicitly. Not applicable as written — but verifying *why* surfaced a real gap in the opposite direction, fixed below. |

### Ported this pass (1, with regression tests)

| Upstream | Change | Rust site |
|---|---|---|
| #7536 (inverted) | **Stateless Responses requests never asked for the encrypted reasoning item.** Upstream's Responses client appends `reasoning.encrypted_content` to `include` whenever a request carries no service-side-storage indicator (`_chat_client.py:1414-1418`); #7536 is Foundry opting *out* of that default. This port set `include` nowhere at all, so it never opted *in*. That quietly defeated machinery it already had: `messages_to_input` re-emits a reasoning item verbatim for a `store: false` tool-loop replay, and drops one that lacks `id`/`encrypted_content` as having "no valid input form" — but the item could never carry `encrypted_content`, because the request never asked for it. `responses_include` now builds the array once, shared by both Responses clients, and Foundry turns the implicit add off via `AzureOpenAIResponsesClient::without_implicit_encrypted_reasoning`, which is #7536's behavior. | `openai/responses.rs` (`responses_include`, `ENCRYPTED_REASONING_INCLUDE`, `build_body`), `azure/responses.rs` (`build_body`, the new builder), `foundry/lib.rs` (both constructors) |

Semantics mirror upstream exactly: a caller's own `include` entries are always
preserved; an explicitly named `reasoning.encrypted_content` is honored even
with the implicit add disabled (the switch governs only what is added
unprompted) and is never duplicated; the trigger is the service-side-storage
indicator rather than `store`; and an empty `include` is omitted rather than
sent as `[]`.

Verified: full workspace build, `cargo test --workspace --all-features`
(**1614 passing**, 8 of them new), `cargo clippy --all-targets --all-features`
clean, `cargo fmt --check` clean. The Foundry opt-out is asserted on the real
outbound body through the hermetic loopback server, not on the flag, and was
confirmed to fail without the wiring.

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
