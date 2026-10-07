//! The parity ledger: `docs/parity/ledger.json`, joined against the .NET
//! declaration inventory and the Go team's .NET→Go mapping (both vendored
//! under `docs/parity/upstream/`) and the Rust index from [`crate::index`].

use crate::index::{self, RustIndex};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

const LEDGER: &str = "docs/parity/ledger.json";
const INVENTORY: &str = "docs/parity/upstream/dotnet-sdk-symbol-inventory.json";
const GO_MAPPING: &str = "docs/parity/upstream/dotnet-go-sdk-symbol-mapping.json";
const REPORT: &str = "docs/parity/STATUS.md";

/// Member groups, in report order. `type` is the type-level `mapping` leaf.
const GROUPS: &[&str] = &[
    "type",
    "constructors",
    "properties",
    "methods",
    "fields",
    "constants",
    "events",
];
const STATUSES: &[&str] = &["mapped", "adapted", "partial", "unmapped", "intentional"];

/// One assessed declaration: `namespace`, declaring type, group, member key.
type Key = (String, String, String, String);

#[derive(Clone)]
struct Leaf {
    status: String,
    note: String,
    rust: Vec<String>,
    area: String,
}

struct Inputs {
    ledger: Value,
    rust: BTreeMap<Key, Leaf>,
    go: BTreeMap<Key, Leaf>,
    inventory: Value,
}

fn read_json(root: &Path, rel: &str) -> Result<Value, String> {
    let text = fs::read_to_string(root.join(rel)).map_err(|e| format!("reading {rel}: {e}"))?;
    serde_json::from_str(&text).map_err(|e| format!("{rel}: {e}"))
}

/// Flattens a namespace → type → group → member tree into leaves. `symbols`
/// names the field holding counterparts (`rust` here, `go_symbols` in Go's).
fn flatten(tree: &Value, symbols: &str, errors: &mut Vec<String>) -> BTreeMap<Key, Leaf> {
    let mut out = BTreeMap::new();
    let empty = Map::new();
    for (ns, types) in tree["namespaces"].as_object().unwrap_or(&empty) {
        for (ty, entry) in types.as_object().unwrap_or(&empty) {
            let area = entry["area"].as_str().unwrap_or_default().to_string();
            let mut push = |group: &str, member: &str, leaf: &Value| {
                let key = (
                    ns.clone(),
                    ty.clone(),
                    group.to_string(),
                    member.to_string(),
                );
                let rust = leaf[symbols]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|s| s.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default();
                let parsed = Leaf {
                    status: leaf["status"].as_str().unwrap_or_default().to_string(),
                    note: leaf["note"].as_str().unwrap_or_default().to_string(),
                    rust,
                    area: area.clone(),
                };
                out.insert(key, parsed);
            };
            if let Some(m) = entry.get("mapping") {
                push("type", "", m);
            }
            for (group, members) in entry.as_object().unwrap_or(&empty) {
                if matches!(group.as_str(), "area" | "assembly" | "mapping") {
                    continue;
                }
                if !GROUPS.contains(&group.as_str()) {
                    errors.push(format!("{ns}::{ty}: unknown group `{group}`"));
                    continue;
                }
                for (member, leaf) in members.as_object().unwrap_or(&empty) {
                    push(group, member, leaf);
                }
            }
        }
    }
    out
}

fn load(root: &Path, errors: &mut Vec<String>) -> Result<Inputs, String> {
    let ledger = read_json(root, LEDGER)?;
    let go_tree = read_json(root, GO_MAPPING)?;
    let inventory = read_json(root, INVENTORY)?;
    let rust = flatten(&ledger, "rust", errors);
    let go = flatten(&go_tree, "go_symbols", &mut Vec::new());
    Ok(Inputs {
        ledger,
        rust,
        go,
        inventory,
    })
}

// --- resolving a ledger key against the .NET inventory -------------------

/// `AIContextProvider.InvokedContext` → `AIContextProvider+InvokedContext`,
/// `AgentResponse<T>` → ``AgentResponse`1``: the catalog's source-like type
/// key in the inventory's CLR spelling.
fn clr_type_key(ns: &str, ty: &str) -> String {
    let mut segments = Vec::new();
    let mut depth = 0usize;
    let mut current = String::new();
    for c in ty.chars() {
        match c {
            '<' => {
                depth += 1;
                current.push(c);
            }
            '>' => {
                depth = depth.saturating_sub(1);
                current.push(c);
            }
            '.' if depth == 0 => segments.push(std::mem::take(&mut current)),
            _ => current.push(c),
        }
    }
    segments.push(current);
    let segments: Vec<String> = segments
        .into_iter()
        .map(|s| match s.split_once('<') {
            Some((name, args)) => format!("{name}`{}", top_level_count(args.trim_end_matches('>'))),
            None => s,
        })
        .collect();
    format!("{ns}.{}", segments.join("+"))
}

/// Number of comma-separated entries at nesting depth zero.
fn top_level_count(list: &str) -> usize {
    if list.trim().is_empty() {
        return 0;
    }
    let mut depth = 0i32;
    let mut count = 1;
    for c in list.chars() {
        match c {
            '<' | '(' | '[' => depth += 1,
            '>' | ')' | ']' => depth -= 1,
            ',' if depth == 0 => count += 1,
            _ => {}
        }
    }
    count
}

/// `(base name, parameter count or None for a property-like key)`.
fn member_shape(key: &str) -> (String, Option<usize>) {
    let head = key.split(" -> ").next().unwrap_or(key);
    let (name, params) = match head.split_once('(') {
        Some((name, rest)) => (name, Some(top_level_count(rest.trim_end_matches(')')))),
        None => (head, None),
    };
    let name = name.split(['<', '`']).next().unwrap_or(name).trim();
    (name.to_string(), params)
}

fn resolves_in_inventory(inventory: &Value, key: &Key) -> bool {
    let (ns, ty, group, member) = key;
    let Some(entry) = inventory["types"].get(clr_type_key(ns, ty)) else {
        return false;
    };
    if group == "type" {
        return true;
    }
    let (mut name, arity) = member_shape(member);
    if group == "constructors" {
        name = ".ctor".to_string();
    }
    entry[group.as_str()].as_object().is_some_and(|members| {
        members.keys().any(|k| {
            let (n, a) = member_shape(k);
            n == name && (arity.is_none() || a.is_none() || a == arity)
        })
    })
}

fn label(key: &Key) -> String {
    let (ns, ty, group, member) = key;
    if group == "type" {
        format!("{ns}::{ty}")
    } else {
        format!("{ns}::{ty} {group}::{member}")
    }
}

// --- commands -------------------------------------------------------------

pub fn check(root: &Path, build: bool) -> Result<(), String> {
    let rust_index = if build {
        index::build(root)?
    } else {
        index::load(root)?
    };
    let mut errors = Vec::new();
    let inputs = load(root, &mut errors)?;
    validate(&inputs, &rust_index, &mut errors);

    let expected = render_report(&inputs);
    let current = fs::read_to_string(root.join(REPORT)).unwrap_or_default();
    if current != expected {
        errors.push(format!(
            "{REPORT} is out of date; run `cargo xtask parity report`"
        ));
    }

    if errors.is_empty() {
        println!(
            "parity ledger OK: {} assessed declarations, {} Rust symbols indexed",
            inputs.rust.len(),
            rust_index.len()
        );
        Ok(())
    } else {
        for e in &errors {
            eprintln!("  {e}");
        }
        Err(format!("{} problem(s) in the parity ledger", errors.len()))
    }
}

fn validate(inputs: &Inputs, rust_index: &RustIndex, errors: &mut Vec<String>) {
    if inputs.ledger["schema_version"].as_u64() != Some(1) {
        errors.push("ledger schema_version must be 1".into());
    }
    for (key, leaf) in &inputs.rust {
        let at = label(key);
        if !STATUSES.contains(&leaf.status.as_str()) {
            errors.push(format!("{at}: unknown status `{}`", leaf.status));
        }
        if leaf.note.trim().is_empty() {
            errors.push(format!("{at}: a note is required"));
        }
        let needs_symbols = matches!(leaf.status.as_str(), "mapped" | "adapted" | "partial");
        if needs_symbols && leaf.rust.is_empty() {
            errors.push(format!(
                "{at}: `{}` needs at least one Rust symbol",
                leaf.status
            ));
        }
        if !needs_symbols && !leaf.rust.is_empty() {
            errors.push(format!("{at}: `{}` takes no Rust symbols", leaf.status));
        }
        if leaf.status == "intentional" && !leaf.note.contains("SCOPE.md") {
            errors.push(format!(
                "{at}: an `intentional` note must cite the SCOPE.md entry that decides it"
            ));
        }
        let type_key = (
            key.0.clone(),
            key.1.clone(),
            "type".to_string(),
            String::new(),
        );
        let area = if leaf.area.is_empty() {
            inputs
                .go
                .get(&type_key)
                .map(|l| l.area.as_str())
                .unwrap_or_default()
        } else {
            &leaf.area
        };
        let go_has_type = inputs.go.keys().any(|k| k.0 == key.0 && k.1 == key.1);
        if area.is_empty() && !go_has_type {
            errors.push(format!(
                "{at}: a type the Go catalog does not list needs an `area`"
            ));
        }
        if !inputs.go.contains_key(key) && !resolves_in_inventory(&inputs.inventory, key) {
            errors.push(format!(
                "{at}: not found in the Go catalog or the .NET inventory (check the spelling)"
            ));
        }
        for symbol in &leaf.rust {
            if !rust_index.contains(symbol) {
                errors.push(format!("{at}: Rust symbol `{symbol}` does not exist"));
            }
        }
    }
}

/// Per-namespace counts: Go statuses over Go's assessed leaves, Rust
/// statuses over the same leaves, and how many Rust has not reviewed.
struct Row {
    go: BTreeMap<String, usize>,
    rust: BTreeMap<String, usize>,
    unreviewed: usize,
    rust_only: usize,
}

fn tally(inputs: &Inputs) -> BTreeMap<String, Row> {
    fn row<'a>(rows: &'a mut BTreeMap<String, Row>, ns: &str) -> &'a mut Row {
        rows.entry(ns.to_string()).or_insert_with(|| Row {
            go: BTreeMap::new(),
            rust: BTreeMap::new(),
            unreviewed: 0,
            rust_only: 0,
        })
    }
    let mut rows = BTreeMap::new();
    for (key, go) in &inputs.go {
        let r = row(&mut rows, &key.0);
        *r.go.entry(go.status.clone()).or_default() += 1;
        match inputs.rust.get(key) {
            Some(rust) => *r.rust.entry(rust.status.clone()).or_default() += 1,
            None => r.unreviewed += 1,
        }
    }
    for (key, rust) in &inputs.rust {
        if !inputs.go.contains_key(key) {
            let r = row(&mut rows, &key.0);
            *r.rust.entry(rust.status.clone()).or_default() += 1;
            r.rust_only += 1;
        }
    }
    rows
}

fn counts(map: &BTreeMap<String, usize>) -> String {
    STATUSES
        .iter()
        .map(|s| map.get(*s).copied().unwrap_or(0).to_string())
        .collect::<Vec<_>>()
        .join(" / ")
}

pub fn summary(root: &Path) -> Result<(), String> {
    let inputs = load(root, &mut Vec::new())?;
    println!("counts are {}", STATUSES.join(" / "));
    for (ns, r) in tally(&inputs) {
        println!(
            "{ns}\n  go:   {}\n  rust: {}  (unreviewed {}, beyond Go's catalog {})",
            counts(&r.go),
            counts(&r.rust),
            r.unreviewed,
            r.rust_only
        );
    }
    Ok(())
}

/// Declarations Go (or .NET alone) has that Rust lacks or only partly has.
fn gap_rows(inputs: &Inputs) -> Vec<(Key, String, String, String)> {
    let mut rows = Vec::new();
    let mut keys: BTreeSet<&Key> = inputs.go.keys().collect();
    keys.extend(inputs.rust.keys());
    for key in keys {
        let go = inputs.go.get(key).map(|l| l.status.as_str()).unwrap_or("-");
        let (rust, note) = match inputs.rust.get(key) {
            Some(l) => (l.status.as_str(), l.note.as_str()),
            None => ("unreviewed", ""),
        };
        let rust_lacks = matches!(rust, "unmapped" | "partial");
        let go_has = matches!(go, "mapped" | "adapted");
        if rust_lacks || (rust == "unreviewed" && go_has) {
            rows.push((
                key.clone(),
                rust.to_string(),
                go.to_string(),
                note.to_string(),
            ));
        }
    }
    rows
}

pub fn gaps(root: &Path, namespace: Option<&str>) -> Result<(), String> {
    let inputs = load(root, &mut Vec::new())?;
    for (key, rust, go, note) in gap_rows(&inputs) {
        if namespace.is_some_and(|ns| !key.0.contains(ns)) {
            continue;
        }
        println!("[rust {rust:10} | go {go:11}] {}", label(&key));
        if !note.is_empty() {
            println!("    {note}");
        }
    }
    Ok(())
}

pub fn write_report(root: &Path) -> Result<(), String> {
    let inputs = load(root, &mut Vec::new())?;
    fs::write(root.join(REPORT), render_report(&inputs))
        .map_err(|e| format!("writing {REPORT}: {e}"))?;
    println!("wrote {REPORT}");
    Ok(())
}

fn render_report(inputs: &Inputs) -> String {
    let b = &inputs.ledger["baseline"];
    let s = |k: &str| b[k].as_str().unwrap_or("?").to_string();
    let mut out = String::new();
    let _ = writeln!(out, "# Parity status\n");
    let _ = writeln!(
        out,
        "Generated by `cargo xtask parity report` from [`ledger.json`](ledger.json); do not edit by hand.\n"
    );
    let _ = writeln!(
        out,
        "Baseline: .NET inventory `{}`, Go catalog at `{}`, Rust `{}`, assessed {}.\n",
        s("dotnet_inventory_release"),
        s("go_commit"),
        s("rust_version"),
        s("checked_at")
    );
    let _ = writeln!(
        out,
        "Each cell counts {}. A count is not a parity percentage: \
         a declaration maps to a Rust symbol, not to equivalent behaviour.\n",
        STATUSES.join(" / ")
    );
    let _ = writeln!(
        out,
        "| Namespace | Go | Rust | Rust unreviewed | Rust beyond Go's catalog |"
    );
    let _ = writeln!(out, "|---|---|---|---|---|");
    let mut total_unreviewed = 0;
    for (ns, r) in tally(inputs) {
        total_unreviewed += r.unreviewed;
        let _ = writeln!(
            out,
            "| `{ns}` | {} | {} | {} | {} |",
            counts(&r.go),
            counts(&r.rust),
            r.unreviewed,
            r.rust_only
        );
    }
    let gaps = gap_rows(inputs);
    let (assessed, unreviewed): (Vec<_>, Vec<_>) = gaps
        .into_iter()
        .partition(|(_, rust, _, _)| rust != "unreviewed");
    let _ = writeln!(out, "\n## Gaps\n");
    let _ = writeln!(
        out,
        "Assessed declarations Rust lacks (`unmapped`) or only partly covers (`partial`).\n"
    );
    if assessed.is_empty() {
        let _ = writeln!(out, "None.");
    } else {
        let _ = writeln!(out, "| Declaration | Rust | Go | Note |");
        let _ = writeln!(out, "|---|---|---|---|");
        for (key, rust, go, note) in &assessed {
            let _ = writeln!(
                out,
                "| `{}` | {rust} | {go} | {} |",
                label(key),
                note.replace('|', "\\|").replace('\n', " ")
            );
        }
    }
    let _ = writeln!(out, "\n## Not yet reviewed\n");
    let _ = writeln!(
        out,
        "{} declarations Go maps that the Rust ledger has not assessed yet ({} unreviewed in all), by type:\n",
        unreviewed.len(),
        total_unreviewed
    );
    let mut by_type: BTreeMap<String, usize> = BTreeMap::new();
    for (key, ..) in &unreviewed {
        *by_type.entry(format!("{}::{}", key.0, key.1)).or_default() += 1;
    }
    for (ty, n) in by_type {
        let _ = writeln!(out, "- `{ty}` ({n})");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_keys_take_the_clr_spelling() {
        assert_eq!(
            clr_type_key("Microsoft.Agents.AI", "AIContextProvider.InvokedContext"),
            "Microsoft.Agents.AI.AIContextProvider+InvokedContext"
        );
        assert_eq!(
            clr_type_key("Microsoft.Agents.AI", "AgentResponse<T>"),
            "Microsoft.Agents.AI.AgentResponse`1"
        );
        assert_eq!(
            clr_type_key("Microsoft.Agents.AI.Workflows", "Executor<TInput, TOutput>"),
            "Microsoft.Agents.AI.Workflows.Executor`2"
        );
    }

    #[test]
    fn member_shapes_ignore_generics_and_return_types() {
        assert_eq!(
            member_shape("RunAsync<T>(string, AgentSession, JsonSerializerOptions, AgentRunOptions, CancellationToken)"),
            ("RunAsync".into(), Some(5))
        );
        assert_eq!(
            member_shape("GetService``1(System.Object) -> !!0"),
            ("GetService".into(), Some(1))
        );
        assert_eq!(
            member_shape("AddEdge(Microsoft.Agents.AI.Workflows.ExecutorBinding,System.Func`2<System.Object,System.Boolean>) -> X"),
            ("AddEdge".into(), Some(2))
        );
        assert_eq!(member_shape("Name -> System.String"), ("Name".into(), None));
        assert_eq!(member_shape("Build()"), ("Build".into(), Some(0)));
    }
}
