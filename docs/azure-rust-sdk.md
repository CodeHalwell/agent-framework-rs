# Official Azure SDK for Rust crates, and what this port could use

A review of what Microsoft actually ships for Rust, measured against what
this port hand-rolls against Azure services. Checked **2026-11-02**, with
versions read from the crates.io API rather than from announcement posts,
which lag.

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

Worth checking before committing to it: `azure_data_cosmos` 1.0.0's query
surface has to cover the `VectorDistance` queries `vector_store.rs` issues,
which are a relatively recent Cosmos feature. If it does not expose them,
scope the feature to `auth` + plain document CRUD and keep issuing vector
queries directly.

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
`opentelemetry` 0.32). The `azure_core_opentelemetry` bridge for the Azure
SDK's own client spans still waits on its move to `opentelemetry` 0.32.

### 3. Blob session store: `azure_storage_blob` is at 1.1.0, not 1.1

`SCOPE.md` already names `azure_storage_blob` 1.1 for the
`AzureBlobAgentSessionStore` analogue, and notes it shares the `azure_core`
1.x line that `entra-sdk` already pulls. That is still true and the version
is now **1.1.0 stable**, so the item is unblocked on the dependency side —
it needs only the `session_store` implementation.

## Not worth doing

- **Key Vault** (`azure_security_keyvault_*`): stable, but nothing in this
  port reads secrets from Key Vault. Configuration comes from the
  environment and credentials from the Entra chain.
- **Event Hubs / Storage Queue**: no surface here needs a message bus.
- **Replacing the hand-rolled credential chain outright.** `entra-sdk` is
  the right shape — optional. The hand-rolled chain has no Azure SDK
  dependency and keeps the default build light, which matters for a library.
