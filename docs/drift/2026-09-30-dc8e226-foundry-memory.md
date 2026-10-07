# Verification pass + the Foundry memory provider (same upstream baseline, `dc8e226`)

A follow-up over the [post-`6606bef` pass](2026-09-28-dc8e226.md): re-check its six changes against primary
sources, then spend the rest on the standing-gap list. The re-check held —
and it turned up that the list's own top entry had been closed off for the
wrong reason.

### What the re-check confirmed

The riskiest call in that pass was encoding an empty hosted-MCP
allowlist as `tool_configuration: {"enabled": false}` rather than upstream's
literal `allowed_tools: []`. Anthropic's MCP connector documentation settles
both halves of it. Its migration table off the deprecated beta reads
`tool_configuration.enabled: false` as a first-class "no tools" and — the
line that matters — describes **"No `tool_configuration` (all tools
enabled)"**, which is precisely the inversion the fix removes. So the
diagnosis was right and the encoding is the documented one, not merely the
defensible one. The other five were re-read against their diffs; the Hamming
normalization's one unguarded edge (`a.len()` as a divisor) is unreachable
because a zero-dimension vector field is rejected at definition validation
(`vectors.rs:383`).

One genuinely new thing came out of that reading, and it is **not** a bug
here: `mcp-client-2025-04-04`, the beta this crate sends, is deprecated in
favour of `mcp-client-2025-11-20`, which moves tool configuration out of
`mcp_servers[].tool_configuration` and into an `mcp_toolset` entry in
`tools[]`. Upstream Python sends the same deprecated flag
(`_chat_client.py`'s `BETA_FLAGS`), so this port is faithful and moving alone
would be a divergence, not a fix. Recorded below with the mapping so that
whoever follows upstream across has it ready.

### The standing-gap list was wrong about why it was stuck

Two Azure items — the Foundry memory provider and Content Understanding —
were recorded as externally blocked on the same premise: the SDKs carrying
their wire contracts "are not available in this environment", so "the REST
paths, the api-version and the long-running-operation shape would all be
guesses."

That premise does not hold: the package index is reachable from this
environment. `azure-ai-projects` and `azure-ai-contentunderstanding` both
download and unpack, and their generated request builders state the contract
outright. The gap was never external; it was an untested assumption, and it
had been carried forward across passes as settled.

### Ported this pass

| Gap | Change | Rust site |
|---|---|---|
| **Foundry managed memory had no provider.** Flagged two passes running as "the most tractable Azure item left", then shelved as unportable. With `azure-ai-projects` readable the contract is explicit: `POST {project_endpoint}/memory_stores/{name}:search_memories` and `:update_memories`, `api-version=v1`, bearer-scoped to `https://ai.azure.com/.default` — already this crate's `FOUNDRY_SCOPE`. Bodies are `{scope, items?, previous_search_id?}` and `{scope, items?, previous_update_id?, update_delay?}` with nulls dropped (the SDK's own `{k: v for k, v in body.items() if v is not None}`); `items` are `{"type":"message","role":…,"content":…}`; the search answers `{search_id, memories[].memory_item.content}`. Three things are faithful rather than invented. The **incremental cursors** only advance on a search that returned something, so an empty answer does not reset where the next one resumes. The **static (user-profile) fetch** runs once per provider and its latch is set even when it *fails*, so an unreachable store costs one request rather than one per run. And **every failure is swallowed and logged**: retrieval and storage are enhancements, and `after_run` also runs on the agent's own failure path, where raising would replace the real error with this one. The one structural divergence is forced by the trait: Python's hooks get a per-run `state` dict and a `SessionContext`, while Rust's `after_run` gets neither — so the cursors, the latch and the session id live in one `Mutex`-guarded struct on the provider, with the session id latched during `before_run` so `after_run` can still resolve a scope. That is the same shape `Mem0Provider` already uses for the same signature gap. Upstream's `begin_update_memories` returns an LRO poller it never polls, reading only `update_id`; the single POST here is exactly that much of the operation. | `foundry/memory.rs` (new), `foundry/lib.rs` |

Ten tests, five of them loopback against a fake data plane on a real socket,
pinning the routes, the bearer header, the null-dropping, both cursors, the
once-only static fetch, and that a 500 leaves the run intact. Both halves of
the risky behaviour were probed by mutation: removing the once-only latch
breaks the cursor test with the right symptom, and propagating the search
error instead of logging it breaks the failure test. Full workspace:
**2093 passing, 0 failing**, clippy `-D warnings`, rustfmt and `cargo doc`
clean.

### Re-triaged, with the reason corrected

| Gap | Status | Assessment |
|---|---|---|
| Foundry — memory provider | ✅ **Closed above** | Was never externally blocked. |
| Azure AI Content Understanding | ❌, unblocked | `azure-ai-contentunderstanding` reads the same way: `/analyzers`, `/analyzers/{analyzer}`, api-versions `2025-11-01` (GA) and `2026-06-01-preview`. What makes it the larger job is size, not mystery — ~1400 lines upstream across a context provider, a file-search backend pair, and content detection. Now the top item with nothing in its way. |
| Foundry — evaluations | ❌ | Same SDK, so also readable now; still large, and upstream is still moving it. The "blocked" half of the old reason is gone; the "moving target" half stands. |
| Anthropic hosted MCP — `mcp-client-2025-11-20` | ❌, deliberate | Not a defect: upstream sends the same deprecated `mcp-client-2025-04-04`, and this port matches it. The mapping when upstream moves: no `tool_configuration` → an `mcp_toolset` with neither `default_config` nor `configs`; `enabled: false` → `default_config.enabled: false`; `allowed_tools: [...]` → `default_config.enabled: false` plus those tools enabled in `configs`. Note the new beta also *requires* the `mcp_toolset` entry — `mcp_servers` alone is rejected — so this is a two-part change, not a rename. |
| Switch/case predicate cannot report failure (#8490) | ✅ (closed in the post-`dc8e226` pass) | Re-examined and declined here, with the blast radius measured: `Condition` returns `bool`, and the `Selection` it feeds returns `Vec<String>`, so neither has an error channel. Making a predicate fallible means widening both public type aliases and every builder that takes one. Worth doing deliberately, not as a side effect of a verification pass. **Done in the `301a43c` pass**, and the measurement above turned out to overstate it: only hand-constructed `Condition`/`Selection` values broke, because the closure-taking APIs became generic over `bool`-or-`Result<bool>`. |
