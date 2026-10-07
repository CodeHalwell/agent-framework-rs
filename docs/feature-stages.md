# Feature stages

Upstream stages its APIs. Python marks a surface with
`@experimental(ExperimentalFeature.X)` and warns on use, while .NET puts it
behind `[Experimental("MAAI001")]`, a compile-time diagnostic. Either way,
the API carries no compatibility promise until upstream says it is stable.

This port does the same with **cargo features**. An experimental surface
compiles only when its `experimental-*` feature is on. Like .NET, that makes
it an explicit opt-in at build time. Nothing behind such a feature is
covered by semver: it can change or disappear in any minor release, as it
can upstream.

## Rules

- **Naming:** a feature is named `experimental-<id>`, where `<id>` is the
  Python `ExperimentalFeature` id in kebab case (`VECTOR_STORES` becomes
  `experimental-vector-stores`). One upstream id gives one feature name,
  whichever crates declare it.
- **Defaults:** an `experimental-*` feature is never in a crate's `default`
  set and never in the umbrella crate's `full`. The umbrella's
  `experimental` feature turns them all on.
- **Forwarding:** a crate whose experimental surface builds on another
  crate's enables that crate's feature of the same name. The umbrella
  crate re-exposes every `experimental-*` feature under the same name.
- **Docs:** each gated item carries `doc(cfg(...))`, so docs.rs (built with
  all features) labels it with the feature it needs.
- **Release candidates:** upstream's release-candidate stage
  (`ReleaseCandidateFeature`, currently empty) stays behind the same
  `experimental-*` gate until upstream declares the surface stable. A
  release candidate can still change, so it carries no promise here either.
- **Graduating:** when upstream drops the marker, remove the feature and its
  `cfg`s in the same PR that ports the change. Note it in the changelog,
  because code that enabled the feature must stop naming it.
- **Enforcement:** `cargo xtask features check` checks these rules against
  every `Cargo.toml` and the table below, and runs in CI.

## Current features

<!-- feature-table: kept in sync by `cargo xtask features check` -->
| Feature | Upstream id | Crates | What it gates |
|---|---|---|---|
| `experimental-vector-stores` | `VECTOR_STORES` | `agent-framework-core`, `agent-framework-cosmos`, `agent-framework-azure-ai-search` | `agent_framework_core::vectors` (stores, collections, portable filters, the vector-collection tools provider), `CosmosVectorStore`, `AzureAISearchStore` |
| `experimental-file-history` | `FILE_HISTORY` | `agent-framework-core` | `history::FileHistoryProvider` |
| `experimental-progressive-tools` | `PROGRESSIVE_TOOLS` | `agent-framework-core` | `middleware::LiveToolList` and `FunctionInvocationContext::{tools, with_tools, add_tools, remove_tools}`. The function-calling loop still uses the live list internally, so behaviour is unchanged; only the public handle is gated. |
| `experimental-harness` | `HARNESS` | `agent-framework-core` | `agent_framework_core::harness`: `LoopAgent` (with `Judge` and `todos_remaining`), `TodoProvider`, `ToolApprovalAgent` and `AgentModeProvider`. File access, file memory, memory, background agents and `create_harness_agent` are not ported. |
| `experimental-declarative-agents` | `DECLARATIVE_AGENTS` | `agent-framework-declarative` | The whole crate, which is empty without it |
<!-- /feature-table -->

## Upstream experimental ids with no gated Rust surface

| Upstream id | Why there is no feature |
|---|---|
| `TO_PROMPT_AGENT` | `FoundryAgent::to_prompt_agent` returns the definition the agent was built from. That is a getter, not upstream's conversion of an arbitrary agent. |
| `AGENT_HOOKS`, `EVALS`, `FIDES`, `COMPUTER_USE`, `FUNCTIONAL_WORKFLOWS`, `MCP_LONG_RUNNING_TASKS`, `MCP_SKILLS`, `SESSION_STORE`, `FOUNDRY_TOOLS`, `FOUNDRY_PREVIEW_TOOLS` | Not ported yet. Whichever lands first adds its `experimental-*` feature here. |
