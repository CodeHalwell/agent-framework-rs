# Official Azure SDK for Rust crates, and what this port could use

A review of what Microsoft actually ships for Rust, measured against what
this port hand-rolls against Azure services. Re-checked **2026-11-16**
(first written 2026-11-02), with versions read from the crates.io API rather
than from announcement posts, which lag.

Since the 2026-11-02 check, **every stable line is unchanged**, so the pins
in `Cargo.toml` are still current. Two things did move, both on the
prerelease lines, and one of them clears a recorded blocker:

- **`azure_core_opentelemetry` 1.1.0-beta.2 is on `opentelemetry` 0.32** —
  the version this workspace already uses. The version mismatch `SCOPE.md`
  records as blocking the bridge for the Azure SDK's own client spans is
  resolved in that beta (1.0.0 stable is still on 0.31). Adopting it today
  means taking a prerelease Azure dependency, which is not this port's
  policy, so the item stays open — but it now waits on a **1.1.0 stable
  release**, not on upstream deciding to move. That is what the next pass
  should check for.
- The remaining betas advanced without changing the picture: `azure_core`
  1.2.0-beta.2, `azure_identity` and `azure_data_cosmos` 1.1.0-beta.2,
  `azure_storage_blob` 1.2.0-beta.1.

One standing open question is also now **answered**: `azure_data_cosmos`
1.0.0's `ContainerClient` exposes `query_items`, which sends arbitrary
Cosmos SQL with parameters. `VectorDistance(...)` is a server-side SQL
function, not a client feature, so the crate covers the queries
`vector_store.rs` issues. Azure AI Search is still the same `0.0.1`
placeholder last published 2025-03-07.

The short version: of the eight Azure-facing crates in this workspace, only
**Cosmos DB** has an official Rust client worth adopting. Azure AI Search,
Foundry / Azure OpenAI, Purview and Copilot Studio have **no official Rust
library at all** — the names on crates.io are either placeholders or
third-party. The hand-rolled REST clients are not a stopgap for those; they
are the only option.

## What Microsoft officially ships

Everything below is published from `github.com/Azure/azure-sdk-for-rust`.
"Stable" means a 1.x release under semver.

| Crate | Latest stable | Also published | Status |
|---|---|---|---|
| `azure_core` | **1.1.0** | `1.2.0-beta.1` | stable |
| `azure_identity` | **1.0.0** | `1.1.0-beta.1` | stable |
| `azure_data_cosmos` | **1.0.0** | `1.1.0-beta.1` | stable |
| `azure_storage_blob` | **1.1.0** | `1.1.0-beta.2` | stable |
| `azure_storage_queue` | 1.x | — | stable |
| `azure_core_opentelemetry` | **1.0.0** | `1.1.0-beta.1` | stable |
| `azure_security_keyvault_secrets` / `_keys` / `_certificates` | 1.x | `1.1.0-beta.1` | stable |
| `azure_messaging_eventhubs` | — | `0.16.0` | pre-1.0 |

Two corrections to what the May 2026 GA announcement said, both confirmed
against crates.io:

- **Cosmos DB has shipped.** The announcement listed `azure_data_cosmos`
  under "what's coming next", expected sometime in 2026. It is now at
  **1.0.0 stable** (with `1.1.0-beta.1` in flight), last published
  2026-10-22, with ~2.7M downloads. Anything citing that post as evidence
  Cosmos is pre-GA is six months stale.
- **`azure_core_opentelemetry` is stable** at **1.0.0**. The announcement
  mentioned it only in prose, without a version, and it was not in the
  stable table.

The `azure_mgmt_*` crates are unofficial generated control-plane crates and
are unmaintained; they are not part of this.

## What does *not* exist, despite the crate name existing

This is the part that matters most, because a name on crates.io looks like a
library until you check it.

| Service | Crate name | Reality |
|---|---|---|
| Azure AI Search | `azure_search_documents` | **0.0.1 placeholder.** Published from the official repo in **March 2025** and never updated. Its own description reads "the Azure Cognitive Search client library for developing **.NET** applications", i.e. a copy-paste stub. Not a library. |
| Microsoft Purview | `azure_analytics_purview_catalog`, `…_account`, `…_administration`, `…_scanning` | **0.0.1 placeholders**, same official repo, same March 2025, never updated. |
| Azure AI Foundry | `azure_ai_foundry_agents`, `azure_ai_foundry_models`, `azure_ai_foundry_core` | **Third party**, not Microsoft: published from `github.com/bzsanti/azure-ai-foundry`, 0.8.0, last updated 2026-03, ~140–200 downloads each. Not a dependency to take. |
| Azure OpenAI (data plane) | — | No official crate. `azure-openai-rs` and similar are third-party. |
| Copilot Studio | — | No official crate, official or otherwise. |

So for four of this workspace's Azure crates the hand-rolled client is not a
shortcut taken for convenience — there is nothing to adopt. That is worth
stating in the README the next time someone asks why this port talks REST to
Foundry.

## Mapped against this workspace

| This crate | Talks to | Official Rust client? | Position |
|---|---|---|---|
| `agent-framework-azure` | Azure OpenAI (Responses/Chat) | none for the data plane; `azure_core` + `azure_identity` for auth | Already uses both, behind the optional `entra-sdk` feature. Pins are current (`azure_core` 1.1, `azure_identity` 1.0). |
| `agent-framework-foundry` | Azure AI Foundry | none (official); third-party only | Hand-rolled, correctly. |
| `agent-framework-cosmos` | Cosmos DB NoSQL | **`azure_data_cosmos` 1.0.0** | The one real adoption candidate — see below. |
| `agent-framework-azure-ai-search` | Azure AI Search | placeholder only | Hand-rolled, correctly. |
| `agent-framework-purview` | Purview | placeholder only | Hand-rolled, correctly. |
| `agent-framework-copilotstudio` | Copilot Studio | none | Hand-rolled, correctly. |
| `agent-framework-foundry-local` | Foundry Local (localhost) | n/a — local HTTP | Hand-rolled, correctly. |
| `agent-framework-core` (`otel-export`) | OTLP → App Insights | **`azure_core_opentelemetry` 1.0.0** | Candidate for the App Insights gap — see below. |

## Three things worth doing

### 1. Cosmos: adopt `azure_data_cosmos` behind a feature, as `entra-sdk` was

`agent-framework-cosmos` is 5,222 lines, and roughly **1,540 of them**
(`client.rs` 1,164 + `auth.rs` 377) are REST plumbing and the HMAC-SHA256
resource-token signing that `azure_data_cosmos` 1.0.0 now does: request
construction, continuation tokens, partition-key routing, throttling and
the `master key` auth dance. The remaining ~3,700 lines are this port's own
abstractions (`vector_store`, `chat_message_store`, `checkpoint_storage`)
and would stay.

The precedent is already set: `entra-sdk` adds the official credential chain
as a purely additive optional feature while the hand-rolled one stays the
default. The same shape — a `cosmos-sdk` feature — gets the official client's
retry and auth behaviour without a breaking change or a forced dependency.

**Checked 2026-11-16:** the query surface is not a constraint.
`ContainerClient::query_items` takes arbitrary Cosmos SQL with parameters,
and `VectorDistance(...)` is evaluated server-side, so the vector queries
`vector_store.rs` issues need no dedicated client support. The feature can
cover auth, CRUD *and* the vector queries.

### 2. App Insights: `azure_core_opentelemetry` 1.0.0 closes a recorded gap

`SCOPE.md` lists "the dedicated Application Insights exporter" as unported
Azure work. `azure_core_opentelemetry` is now stable at 1.0.0 and is
Microsoft's supported bridge from `azure_core` to OTel. Combined with the
`microsoft.agent-framework` resource attribute this pass added to
`otel-export`'s pipeline, that is most of what the gap needs: the attribute
is what Azure's agent views key off, and the bridge is what carries
`azure_core`'s own spans alongside this crate's.

**Update:** the exporter half is now closed directly.
`agent-framework-core`'s `azure-monitor` feature adds
`OtelExport::with_azure_monitor` (and `azure_monitor_from_env`, reading
`APPLICATIONINSIGHTS_CONNECTION_STRING`), which exports traces and metrics
to Application Insights through the community
`opentelemetry-application-insights` exporter (0.45, the release on
`opentelemetry` 0.32).

**Checked 2026-11-16:** the `azure_core_opentelemetry` bridge for the Azure
SDK's own client spans has made its move — **1.1.0-beta.2 depends on
`opentelemetry` 0.32 and `opentelemetry_sdk` 0.32.1**, matching this
workspace. 1.0.0 stable remains on 0.31, so taking the bridge now means
taking a prerelease; the item is unblocked on substance and waits only on
1.1.0 going stable.

### 3. Blob session store: done

`SCOPE.md` named `azure_storage_blob` 1.1 for the
`AzureBlobAgentSessionStore` analogue. That store is now ported —
`agent-framework-azure`'s `session-store` feature — so this item is closed.
The dependency line is still right: `azure_storage_blob` 1.1.0 stable
(1.2.0-beta.1 in flight) shares the `azure_core` 1.x line `entra-sdk`
already pulls.

## Not worth doing

- **Key Vault** (`azure_security_keyvault_*`): stable, but nothing in this
  port reads secrets from Key Vault. Configuration comes from the
  environment and credentials from the Entra chain.
- **Event Hubs / Storage Queue**: no surface here needs a message bus.
- **Replacing the hand-rolled credential chain outright.** `entra-sdk` is
  the right shape — optional. The hand-rolled chain has no Azure SDK
  dependency and keeps the default build light, which matters for a library.
