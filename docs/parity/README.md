# Parity ledger

A symbol-level record of how this port's public Rust API corresponds to the
official Microsoft Agent Framework, with the Go SDK alongside. The current
counts and gap list are in [`STATUS.md`](STATUS.md), which is generated.

The design follows the Go team's
[.NET and Go SDK symbol mapping](https://github.com/microsoft/agent-framework-go/blob/main/docs/dotnet-go-sdk-feature-comparison.md),
and it reuses their data directly:

- **.NET** is the reference, through the declaration inventory the Go
  repository extracts from the published NuGet packages.
- **Go** comes from the Go team's own reviewed mapping, so every row shows
  how Go assessed the same declaration.
- **Rust** is [`ledger.json`](ledger.json), which this repository
  maintains.

Both upstream files are vendored unmodified under [`upstream/`](upstream/SOURCE.md).

Deliberate exclusions are decided in [`SCOPE.md`](../../SCOPE.md). The
ledger records them, but does not decide them.

## Commands

```text
cargo xtask parity check     # rebuild the Rust index, validate the ledger
cargo xtask parity summary   # counts per namespace, Rust beside Go
cargo xtask parity gaps      # declarations Rust lacks or only partly has
cargo xtask parity gaps Workflows   # the same, filtered by namespace
cargo xtask parity report    # regenerate STATUS.md
cargo xtask parity index     # rebuild only the Rust index
```

`check` runs in CI. It fails when:

- a ledger entry names a Rust symbol that does not exist (so a rename or
  removal cannot silently orphan a mapping);
- an entry's .NET key resolves in neither the Go catalog nor the .NET
  inventory;
- a status or note breaks the rules below;
- `STATUS.md` is stale.

The Rust index comes from rustdoc's JSON output for every crate under
`crates/`, with all features enabled. rustdoc JSON is unstable, so the tool
sets `RUSTC_BOOTSTRAP=1` on the stable toolchain rather than requiring
nightly. The index lands in `target/parity/rust-index.json`.

## Format

`ledger.json` mirrors the Go catalog's shape: namespace, then the .NET
type's short name, then member groups, then member keys. Keys are spelled as
the Go catalog spells them: source-like signatures without parameter names,
such as `RunAsync(string, AgentSession, AgentRunOptions, CancellationToken)`.
A nested type uses a dot (`AIContextProvider.InvokingContext`) and a generic
type uses its parameters (`AgentResponse<T>`).

```json
"Microsoft.Agents.AI": {
  "AIAgent": {
    "mapping": {
      "rust": ["agent_framework_core::agent::SupportsAgentRun"],
      "status": "adapted",
      "note": "The abstract base class becomes a trait."
    },
    "methods": {
      "RunAsync(string, AgentSession, AgentRunOptions, CancellationToken)": {
        "rust": ["agent_framework_core::agent::Agent::run"],
        "status": "adapted",
        "note": "Rust cancels by dropping the future, so there is no token parameter."
      }
    }
  }
}
```

- `mapping` is the type-level leaf. The other groups are `constructors`,
  `properties`, `methods`, `fields`, `constants` and `events`.
- `area` is needed only on a type the Go catalog does not list.
- Rust symbols use the item's canonical path, such as
  `agent_framework_core::session::AgentSession`, plus `::member` for a
  field, variant, method or associated item. Re-exports through the
  `agent-framework` umbrella crate are not separate symbols.
- An optional `python` string may name the Python counterpart where that
  helps a reader.

### Statuses

These mean the same as in the Go catalog.

| Status | Meaning | Rust symbols |
|---|---|---|
| `mapped` | A direct counterpart exists. Naming and language differences are fine. | required |
| `adapted` | Covered by a different Rust shape, such as a builder, trait, method pair, `Option` field or middleware. Not a gap. | required |
| `partial` | A counterpart exists but lacks something specific, which the note names. | required |
| `unmapped` | No counterpart yet. | none |
| `intentional` | No counterpart by decision. The note must cite the `SCOPE.md` entry. | none |

Every leaf needs a one-sentence note. A mapping records a counterpart, not
equivalent behaviour, and the counts in `STATUS.md` are not a parity
percentage.

## Keeping it current

- **Changing a mapped Rust API:** `check` fails until the ledger is updated
  in the same PR.
- **Porting a gap:** flip its leaf from `unmapped` to `mapped` or `adapted`,
  then run `cargo xtask parity report`.
- **Refreshing upstream:** follow [`upstream/SOURCE.md`](upstream/SOURCE.md).
  New .NET declarations appear under "Not yet reviewed" in `STATUS.md`.
- **The weekly drift pass** ([`docs/drift/`](../drift/README.md)) should
  touch the ledger whenever a ported change adds or renames a public symbol.
