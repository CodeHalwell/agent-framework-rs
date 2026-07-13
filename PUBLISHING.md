# Publishing to crates.io

The workspace is set up so every crate under `crates/` is publishable:

- All inter-crate dependencies are declared in the root
  `[workspace.dependencies]` with **both** `path` and `version`, so published
  packages resolve their siblings from crates.io.
- Every crate inherits `version`, `edition`, `rust-version`, `license`,
  `repository`, `authors`, `keywords`, `categories`, and `readme` (the
  repository-root `README.md`) from `[workspace.package]`.
- The `examples` crate is `publish = false` and is never published.

Packaging is verified: `cargo package -p agent-framework-core` builds the
packaged crate cleanly and includes `README.md`.

## Publish order

Crates must land on crates.io in dependency order (a `cargo publish` fails
while any of its dependencies is missing from the registry). Within a tier
the order does not matter:

1. **Tier 0** — `agent-framework-core`
2. **Tier 1** — `agent-framework-a2a`, `agent-framework-bedrock`,
   `agent-framework-copilotstudio`, `agent-framework-cosmos`,
   `agent-framework-declarative`, `agent-framework-gemini`,
   `agent-framework-hosting`, `agent-framework-mcp`, `agent-framework-mem0`,
   `agent-framework-openai`, `agent-framework-purview`,
   `agent-framework-redis`
3. **Tier 2** — `agent-framework-azure`, `agent-framework-foundry-local`,
   `agent-framework-github-copilot`, `agent-framework-mistral`,
   `agent-framework-ollama` (need `openai`)
4. **Tier 3** — `agent-framework-anthropic` (needs `azure` + `bedrock`),
   `agent-framework-azure-ai-search`, `agent-framework-foundry`
   (need `azure`)
5. **Tier 4** — `agent-framework` (the umbrella crate)

```bash
# One tier at a time; crates.io indexing is fast but not instant, so give
# each tier a moment before publishing the next.
cargo publish -p agent-framework-core
for c in a2a bedrock copilotstudio cosmos declarative gemini hosting mcp mem0 openai purview redis; do
  cargo publish -p agent-framework-$c
done
for c in azure foundry-local github-copilot mistral ollama; do
  cargo publish -p agent-framework-$c
done
for c in anthropic azure-ai-search foundry; do
  cargo publish -p agent-framework-$c
done
cargo publish -p agent-framework
```

## Before the first publish

- **Check the names are free.** `agent-framework*` names may be taken or
  reserved on crates.io; search first and pick a prefix (e.g. `maf-rs-*`) if
  needed — a rename only touches `[package] name`, the workspace dependency
  table, and `use`/doc references.
- **Trademark note.** This is an independent port, not affiliated with or
  endorsed by Microsoft; keep the crate descriptions/README saying so.
- `cargo publish` requires a crates.io API token (`cargo login`).
- Consider `cargo publish --dry-run -p agent-framework-core` as a final
  smoke check; dry runs of dependent crates fail by design until their
  dependencies are actually published.
