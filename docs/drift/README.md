# Upstream drift log

One file per drift pass, newest first. Each pass diffs upstream
[`microsoft/agent-framework`](https://github.com/microsoft/agent-framework)
from the previous baseline to a new one and records what was ported, what
was already satisfied, and what does not apply here. Every item recorded as
landed was verified (full workspace build, `cargo test`, clippy
`--all-targets` and rustfmt, all green) before commit.

**Current upstream baseline: `301a43c` (2026-10-05).**

Deliberate exclusions live in [`SCOPE.md`](../../SCOPE.md); cite it rather
than re-arguing a boundary in a new pass. The July re-baseline's catalogue
of breaking changes is [`UPSTREAM_DRIFT.md`](../../UPSTREAM_DRIFT.md).

## Adding a pass

Create `YYYY-MM-DD-<upstream-sha>.md` (the date and short SHA of the new
baseline) with the pass's heading as its `#` title, add a row to the top of
the table below, and update the baseline line above. A follow-up against an
unchanged baseline takes the same SHA plus a short suffix.

## Passes

| File | Pass |
|---|---|
| [`2026-10-05-301a43c`](2026-10-05-301a43c.md) | Post-`dc8e226` drift + Azure-ecosystem review (checked against `301a43c`, 2026-10-05) |
| [`2026-09-30-dc8e226-hosting-tool-calls`](2026-09-30-dc8e226-hosting-tool-calls.md) | Tool-call serialization on both hosting surfaces (same upstream baseline, `dc8e226`) |
| [`2026-09-30-dc8e226-pr28-review`](2026-09-30-dc8e226-pr28-review.md) | Review round on PR #28 (same upstream baseline, `dc8e226`) |
| [`2026-09-30-dc8e226-foundry-memory`](2026-09-30-dc8e226-foundry-memory.md) | Verification pass + the Foundry memory provider (same upstream baseline, `dc8e226`) |
| [`2026-09-28-dc8e226`](2026-09-28-dc8e226.md) | Post-`6606bef` drift + Azure-ecosystem review (checked against `dc8e226`, 2026-09-28) |
| [`2026-09-21-6606bef`](2026-09-21-6606bef.md) | Post-`061dc28` drift + Azure-ecosystem review (checked against `6606bef`, 2026-09-21) |
| [`2026-09-14-061dc28`](2026-09-14-061dc28.md) | Post-`010a43a` drift + Azure-ecosystem review (checked against `061dc28`, 2026-09-14) |
| [`2026-09-07-010a43a`](2026-09-07-010a43a.md) | Post-`b5d9f4b` drift + Azure-ecosystem review (checked against `010a43a`, 2026-09-07) |
| [`2026-08-29-d8d07eb`](2026-08-29-d8d07eb.md) | Post-`e6d8d99` drift (checked against `d8d07eb`, 2026-08-29) |
| [`2026-08-26-e6d8d99`](2026-08-26-e6d8d99.md) | Post-`a63d462` drift (checked against `e6d8d99`, 2026-08-26) |
| [`2026-08-24-a63d462`](2026-08-24-a63d462.md) | Post-`e1326eb` drift (checked against `a63d462`, 2026-08-24) |
| [`2026-08-20-e1326eb`](2026-08-20-e1326eb.md) | Post-`5c06755` drift (checked against `e1326eb`, 2026-08-20) |
| [`2026-08-16-5c06755`](2026-08-16-5c06755.md) | Post-`2eb8fbb` drift (checked against `5c06755`, 2026-08-16) |
| [`2026-08-11-2eb8fbb`](2026-08-11-2eb8fbb.md) | Post-`266206e` drift (checked against `2eb8fbb`, 2026-08-11) |
| [`2026-08-07-266206e`](2026-08-07-266206e.md) | Post-`4b1afd90` drift (checked against `266206e`, 2026-08-07) |
| [`2026-08-07-4b1afd90`](2026-08-07-4b1afd90.md) | Post-`beb65b21` drift (checked against `4b1afd90`, 2026-08-07) |
| [`2026-07-13-beb65b21`](2026-07-13-beb65b21.md) | Post-`68136ee` drift (checked against `beb65b21`, 2026-07-13) |
| [`2026-07-68136ee-rebaseline`](2026-07-68136ee-rebaseline.md) | The `68136ee` re-baseline: what landed and what remained (2026-07) |
