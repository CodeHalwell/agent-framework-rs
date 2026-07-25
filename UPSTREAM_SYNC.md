# Upstream sync log

The disposition of every upstream change reviewed for a parity sync, so the
next sync starts from a known baseline instead of rediscovering one. The
stale-baseline problem that prompted [UPSTREAM_DRIFT.md](UPSTREAM_DRIFT.md)
— a port whose conclusions were drawn against a seven-month-old checkout —
is the thing this file exists to prevent.

Each entry records what upstream changed, whether it lands anywhere in this
port, and *why* when the answer is no. "Already at parity" entries matter as
much as ported ones: they are the ones a future sync would otherwise
re-litigate.

---

## Sync: `beb65b21` → `c6442de` (2026-07-13 → 2026-07-24)

**Baseline before:** `beb65b21` (2026-07-13), the pin for releases 0.1.0 and
0.1.1.
**Baseline after:** `c6442de` (2026-07-24).
**Commits in window:** 102, all triaged.

Roughly half the window is .NET-only work (`.NET:`-prefixed commits),
dependency bumps, CI changes and samples. Those are listed under
[Out of scope](#out-of-scope) rather than enumerated individually. What
follows is every Python-side change that could plausibly touch this port.

### Ported

| Upstream | Change | Rust site |
|---|---|---|
| #6990 `1c00827` | Structured-value text read from the last non-empty assistant message, joined with no separator | `core/src/types/response.rs` — new `structured_output_text`, used by `ChatResponse::{parse_json,try_parse_value}` and `AgentResponse::parse_json` |
| #7267 `d98ac29` | Approval-replay duplicate check widened from per-message to conversation-wide, excluding already-answered call ids | `core/src/client.rs` — `replace_approval_contents_with_results` |
| #7105 `3604ba7` | Finish-reason normalization: `incomplete_details` precedence, no reason for non-terminal responses, `response.incomplete` treated as terminal; OTel gated to convention values | `openai/src/responses.rs` (Azure + Foundry inherit by delegation), `core/src/observability.rs` — new `otel_finish_reason` |
| #7095 `83ba938` | Gemini 3 `thoughtSignature` preserved across function-call replays | `core/src/types/content.rs` — new `TextReasoningContent::protected_data`; `gemini/src/convert.rs` — parse + replay |

### Already at parity — verified, no change needed

These upstream fixes correct behavior this port never had. Each was checked
against the Rust source rather than assumed.

| Upstream | Change | Why Rust was already correct |
|---|---|---|
| #6809 `df19800` | Function-call name dropped when only the later streaming delta carried it | `FunctionCallContent::merge` already takes `other.name` when `self.name` is empty, and additionally guards against a repeated full name concatenating into `get_weatherget_weather` |
| #7060 `b5300fe` | Per-run `additional_beta_flags` leaking into raw Anthropic request kwargs | `anthropic/src/convert.rs::compute_beta_flags` already removes the key from `additional_properties` while merging it into the header; covered by `compute_beta_flags_does_not_leak_into_request_body` |
| #6297 `f1ba16e` | Magentic manager reused one accumulating session, duplicating the conversation every round | `StandardMagenticManager::complete` calls `agent.run(messages, None)` — it never held a session, which is the behavior upstream moved to |
| #7219 `afdf8af` | Compaction could emit an empty projection | `TokenBudget` always retains the newest non-system message ("compaction never reduces a non-empty tail to nothing"); `Truncation` retains `max_messages` |
| #7124 `7c6b1e9` | Compaction token counts inflated by `\uXXXX` escapes on non-ASCII text | Rust counts tokens over the raw text of each content item; there is no JSON-escaping step to inflate |
| #7108 `0d29250` | Pydantic `model_dump(exclude_none=True)` stripped arguments the model explicitly set to `null` | Arguments are carried as `serde_json::Value` and passed through without a revalidation/re-dump step, so an explicit `null` survives |
| #7200 `fb38b1d` | `PropertySchema.to_json_schema()` not recursing into nested array items / object properties | `declarative/src/agent.rs::PropertySpec::to_json_schema` already recurses into both `items` and nested `properties`, hoisting nested `required` |

### Not applicable

| Upstream | Change | Why it has no Rust site |
|---|---|---|
| #6822 `a4e4a5a` | Ollama parallel tool calls colliding because the function name was used as the call id | This port's Ollama client speaks Ollama's OpenAI-compatible endpoint, which supplies real per-call ids; `convert.rs` already resolves them across streaming chunks by index |
| #6916 `6afae2f` | `ValueError` for malformed data URIs in `detect_media_type_from_base64` | There is no Rust counterpart to that helper. `DataContent::from_bytes` *constructs* data URIs; the only parsing is `openai::convert::{strip_data_uri_prefix, data_content_media_type}`, which are deliberately lenient pass-throughs |
| #7300, #7155 | GitHub Copilot input attachments as inline blobs; `GitHubCopilotOptions` forwarded verbatim to `create_session` | Both are plumbing for the Python `copilot` SDK's `create_session` / `send_and_wait` signatures. This port's Copilot crate does not wrap that SDK |
| #7218 `a4f02aa` | MCP `header_provider` headers not reaching the streamable-HTTP transport | The bug is a Python `ContextVar`-vs-task-context issue: the transport sends from tasks whose context predates `call_tool`. Rust threads headers through the request explicitly; there is no ambient-context path to miss |

### Deferred

Real gaps, not yet closed. Listed so they are picked up deliberately rather
than rediscovered.

| Upstream | Change | Notes |
|---|---|---|
| #7163 `6180272` | OpenAI prompt cache breakpoints for GPT-5.6 (`prompt_cache_options` request option plus a per-content `prompt_cache_breakpoint`) | New feature, not a bug fix. Needs a request-option surface and a content-level annotation |
| #7189 `9e836f7` | MCP tool-use sampling results | Requires a `ToolUseContent` block in `mcp/src/protocol.rs` before the conversion in `sampling.rs` can return one |
| #7097 `0df184e` | Sub-workflow checkpoint restore preserving sub-workflow state | Substantial upstream refactor (+604/−245 across the runner, runner context and workflow executor). Wants its own change |
| #6579 `18b03ea` | Checkpoint encoding handling | Pairs naturally with the sub-workflow checkpoint work above |
| #7041 `c35a63e` | Cross-session origin attribution on context messages | New capability on sessions + context providers |
| #7234 `a70fe21` | Responses conversation-ID helper | Lives in `hosting-responses`, a package with only partial coverage here |

### Out of scope

- **.NET-only commits.** ~30 in this window (`.NET:`-prefixed), including
  the breaking `HarnessAgent`, `ToolApprovalAgent` and message-injection
  graduations. This port tracks the Python surface; .NET is a reference for
  cross-language naming only.
- **Python packages with no Rust counterpart:** `ag-ui`, `devui`,
  `durabletask`, `chatkit`, `harness`, `hosting-telegram`, `foundry_hosting`,
  `azurefunctions`, `lab`, `monty`. Notably this excludes the AG-UI fixes
  (#7102, #7084, #6905, #6804, #7277) and the harness / `create_harness_agent`
  graduations (#7120, #7093, #7094).
- **Dependency bumps, CI, samples, docs and version-bump commits.**

---

## Open finding, not from this window

`Content::as_text` matches both `Content::Text` **and**
`Content::TextReasoning`, so `Message::text()` includes reasoning text.
Upstream's `Message.text` filters on `content.type == "text"` only. Any
provider that emits reasoning therefore yields a `text()` in this port that
concatenates chain-of-thought with the answer.

This predates the 0.1.1 baseline, so it is not part of this sync. It is
recorded here because the structured-output fix above sits on the same code
path and deliberately does *not* rely on `as_text`: `structured_output_text`
matches `Content::Text` directly. Closing the wider divergence means
changing `Message::text()`, which also feeds compaction token counts and
several provider request builders — worth doing, worth doing on its own.
