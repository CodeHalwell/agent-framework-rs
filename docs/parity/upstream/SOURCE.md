# Vendored upstream data

Both JSON files are copied unmodified from
[microsoft/agent-framework-go](https://github.com/microsoft/agent-framework-go)
`docs/` at commit `21528e2a869036953b37da872dd8f21d4481515c` (2026-10-06),
under that repository's MIT licence ([`LICENSE-agent-framework-go`](LICENSE-agent-framework-go)).

| File | What it is |
|---|---|
| `dotnet-sdk-symbol-inventory.json` | Public declarations extracted from the .NET core NuGet packages (`Microsoft.Agents.AI`, `.Abstractions`, `.Workflows`, release 1.22.0) by the Go repository's `cmd/dotnetsymbols`. |
| `dotnet-go-sdk-symbol-mapping.json` | The Go team's reviewed .NET → Go mapping, one leaf per declaration with a status and note. |

To refresh, copy both files from a newer Go commit, update the commit above
and `baseline` in [`../ledger.json`](../ledger.json), then run
`cargo xtask parity check`. Declarations that disappeared upstream show up
as unresolved ledger keys, and new ones appear under "Not yet reviewed" in
`STATUS.md`.
