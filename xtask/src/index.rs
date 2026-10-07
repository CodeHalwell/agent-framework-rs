//! The Rust side of the ledger: every public symbol the workspace's library
//! crates export, read from rustdoc's JSON output.
//!
//! rustdoc JSON is unstable, so it is produced on the pinned stable toolchain
//! with `RUSTC_BOOTSTRAP=1` rather than a nightly whose format would drift
//! under us. The JSON is walked as untyped `serde_json::Value`, touching only
//! long-standing fields (`index`, `paths`, `crate_id`, and the
//! `struct`/`enum`/`trait`/`impl` inners), so a format-version bump does not
//! break the build.
//!
//! A symbol is `crate::module::Item` for an item and
//! `crate::module::Item::member` for a field, variant, method or associated
//! item, using the item's canonical (defining) path. Re-exports through the
//! `agent-framework` umbrella crate are not separate symbols.

use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub type RustIndex = BTreeSet<String>;

const ITEM_KINDS: &[&str] = &[
    "struct",
    "enum",
    "trait",
    "function",
    "constant",
    "static",
    "type_alias",
    "union",
    "macro",
    "module",
    "trait_alias",
];

/// Workspace library crates, by package name, read from `crates/*/Cargo.toml`.
fn library_crates(root: &Path) -> Result<Vec<String>, String> {
    let mut names = Vec::new();
    let dir = root.join("crates");
    let entries = fs::read_dir(&dir).map_err(|e| format!("reading {}: {e}", dir.display()))?;
    for entry in entries {
        let manifest = entry.map_err(|e| e.to_string())?.path().join("Cargo.toml");
        let Ok(text) = fs::read_to_string(&manifest) else {
            continue;
        };
        let name = text
            .lines()
            .find_map(|l| {
                let l = l.trim();
                l.strip_prefix("name")
                    .map(str::trim_start)
                    .and_then(|r| r.strip_prefix('='))
                    .map(|r| r.trim().trim_matches('"').to_string())
            })
            .ok_or_else(|| format!("no package name in {}", manifest.display()))?;
        names.push(name);
    }
    names.sort();
    Ok(names)
}

fn target_dir(root: &Path) -> PathBuf {
    root.join("target").join("parity")
}

pub fn index_path(root: &Path) -> PathBuf {
    target_dir(root).join("rust-index.json")
}

/// Runs rustdoc over every library crate and writes the combined index.
pub fn build(root: &Path) -> Result<RustIndex, String> {
    let target = target_dir(root);
    let mut index = RustIndex::new();
    for name in library_crates(root)? {
        eprintln!("rustdoc json: {name}");
        let status = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
            .current_dir(root)
            .env("RUSTC_BOOTSTRAP", "1")
            .args([
                "rustdoc",
                "--quiet",
                "--lib",
                "--all-features",
                "--package",
                &name,
            ])
            .arg("--target-dir")
            .arg(&target)
            .args(["--", "-Z", "unstable-options", "--output-format", "json"])
            // Doc-link warnings are `cargo doc`'s business, not the index's.
            .args(["--cap-lints", "allow"])
            .status()
            .map_err(|e| format!("running cargo rustdoc for {name}: {e}"))?;
        if !status.success() {
            return Err(format!("cargo rustdoc failed for {name}"));
        }
        let json = target
            .join("doc")
            .join(format!("{}.json", name.replace('-', "_")));
        let text =
            fs::read_to_string(&json).map_err(|e| format!("reading {}: {e}", json.display()))?;
        let doc: Value =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", json.display()))?;
        collect(&doc, &mut index);
    }
    let out = index_path(root);
    let text = serde_json::to_string_pretty(&index).expect("a set of strings serializes");
    fs::write(&out, text + "\n").map_err(|e| format!("writing {}: {e}", out.display()))?;
    Ok(index)
}

/// Loads the index written by the last [`build`].
pub fn load(root: &Path) -> Result<RustIndex, String> {
    let path = index_path(root);
    let text = fs::read_to_string(&path).map_err(|_| {
        format!(
            "{} is missing; run `cargo xtask parity index` first",
            path.display()
        )
    })?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

/// Public positions of a tuple struct's fields. rustdoc lists every position
/// and writes `null` where a field is private, so a position keeps its
/// source index (`Wrapper(pub A, B, pub C)` gives `0` and `2`).
fn tuple_fields(fields: &Value) -> Vec<String> {
    fields
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
        .filter(|(_, id)| !id.is_null())
        .map(|(position, _)| position.to_string())
        .collect()
}

fn collect(doc: &Value, out: &mut RustIndex) {
    let empty = serde_json::Map::new();
    let index = doc["index"].as_object().unwrap_or(&empty);
    let paths = doc["paths"].as_object().unwrap_or(&empty);
    let item = |id: &Value| -> Option<&Value> {
        let key = match id {
            Value::Number(n) => n.to_string(),
            Value::String(s) => s.clone(),
            _ => return None,
        };
        index.get(&key)
    };
    let names_of = |ids: &Value| -> Vec<String> {
        ids.as_array()
            .into_iter()
            .flatten()
            .filter_map(|id| item(id)?.get("name")?.as_str().map(str::to_string))
            .collect()
    };
    let impl_members = |impls: &Value| -> Vec<String> {
        let mut members = Vec::new();
        for imp in impls.as_array().into_iter().flatten() {
            let Some(inner) = item(imp).and_then(|i| i["inner"].get("impl")) else {
                continue;
            };
            if inner["is_synthetic"].as_bool() == Some(true) || !inner["blanket_impl"].is_null() {
                continue;
            }
            let inherent = inner["trait"].is_null();
            for member in inner["items"].as_array().into_iter().flatten() {
                let Some(m) = item(member) else { continue };
                // Inherent items count only when public; trait items are
                // callable wherever the trait is.
                if inherent && m["visibility"].as_str() != Some("public") {
                    continue;
                }
                if let Some(name) = m["name"].as_str() {
                    members.push(name.to_string());
                }
            }
        }
        members
    };

    for (id, summary) in paths {
        if summary["crate_id"].as_u64() != Some(0) {
            continue;
        }
        let kind = summary["kind"].as_str().unwrap_or_default();
        if !ITEM_KINDS.contains(&kind) {
            continue;
        }
        let Some(segments) = summary["path"].as_array() else {
            continue;
        };
        let path = segments
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join("::");
        let Some(entry) = index.get(id) else {
            continue;
        };
        let inner = &entry["inner"];
        let mut members = Vec::new();
        if let Some(s) = inner.get("struct") {
            if let Some(fields) = s["kind"].get("plain").map(|p| &p["fields"]) {
                members.extend(names_of(fields));
            }
            if let Some(fields) = s["kind"].get("tuple") {
                members.extend(tuple_fields(fields));
            }
            members.extend(impl_members(&s["impls"]));
        } else if let Some(e) = inner.get("enum") {
            members.extend(names_of(&e["variants"]));
            members.extend(impl_members(&e["impls"]));
        } else if let Some(u) = inner.get("union") {
            members.extend(names_of(&u["fields"]));
            members.extend(impl_members(&u["impls"]));
        } else if let Some(t) = inner.get("trait") {
            members.extend(names_of(&t["items"]));
        }
        for m in members {
            out.insert(format!("{path}::{m}"));
        }
        out.insert(path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tuple_fields_keep_their_positions_and_skip_private_ones() {
        let doc = json!({
            "index": {
                "1": {"name": "Wrapper", "inner": {"struct": {
                    "kind": {"tuple": [10, null, 12]},
                    "impls": []
                }}},
                "10": {"name": "0", "visibility": "public", "inner": {"struct_field": {}}},
                "12": {"name": "2", "visibility": "public", "inner": {"struct_field": {}}}
            },
            "paths": {
                "1": {"crate_id": 0, "kind": "struct", "path": ["krate", "Wrapper"]}
            }
        });
        let mut index = RustIndex::new();
        collect(&doc, &mut index);
        let got: Vec<_> = index.iter().map(String::as_str).collect();
        assert_eq!(
            got,
            ["krate::Wrapper", "krate::Wrapper::0", "krate::Wrapper::2"]
        );
    }
}
