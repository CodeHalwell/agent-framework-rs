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
    /// Type-level keys for .NET types that neither catalog lists.
    inventory_only: BTreeSet<Key>,
}

fn read_json(root: &Path, rel: &str) -> Result<Value, String> {
    let text = fs::read_to_string(root.join(rel)).map_err(|e| format!("reading {rel}: {e}"))?;
    serde_json::from_str(&text).map_err(|e| format!("{rel}: {e}"))
}

/// Flattens a namespace → type → group → member tree into leaves. `symbols`
/// names the field holding counterparts (`rust` here, `go_symbols` in Go's).
///
/// `source` names the file in diagnostics. A missing `namespaces`, or any
/// container or leaf that is not a JSON object, is an error rather than an
/// empty container, so a malformed entry cannot vanish before validation.
fn flatten(
    tree: &Value,
    source: &str,
    symbols: &str,
    errors: &mut Vec<String>,
) -> BTreeMap<Key, Leaf> {
    let mut out = BTreeMap::new();
    let Some(namespaces) = object(
        &tree["namespaces"],
        &format!("{source}: `namespaces`"),
        errors,
    ) else {
        return out;
    };
    for (ns, types) in namespaces {
        let Some(types) = object(types, &format!("{source}: namespace {ns}"), errors) else {
            continue;
        };
        for (ty, entry) in types {
            let Some(entry_map) = object(entry, &format!("{source}: {ns}::{ty}"), errors) else {
                continue;
            };
            let area = entry["area"].as_str().unwrap_or_default().to_string();
            let mut push = |group: &str, member: &str, leaf: &Value, errors: &mut Vec<String>| {
                let at = label(&(
                    ns.clone(),
                    ty.clone(),
                    group.to_string(),
                    member.to_string(),
                ));
                if object(leaf, &format!("{source}: {at}"), errors).is_none() {
                    return;
                }
                let key = (
                    ns.clone(),
                    ty.clone(),
                    group.to_string(),
                    member.to_string(),
                );
                let rust = symbol_list(
                    leaf.get(symbols),
                    &format!("{source}: {at}"),
                    symbols,
                    errors,
                );
                let parsed = Leaf {
                    status: leaf["status"].as_str().unwrap_or_default().to_string(),
                    note: leaf["note"].as_str().unwrap_or_default().to_string(),
                    rust,
                    area: area.clone(),
                };
                out.insert(key, parsed);
            };
            if let Some(m) = entry.get("mapping") {
                push("type", "", m, errors);
            }
            for (group, members) in entry_map {
                if matches!(group.as_str(), "area" | "assembly" | "mapping") {
                    continue;
                }
                if !GROUPS.contains(&group.as_str()) {
                    errors.push(format!("{source}: {ns}::{ty}: unknown group `{group}`"));
                    continue;
                }
                let Some(members) =
                    object(members, &format!("{source}: {ns}::{ty} {group}"), errors)
                else {
                    continue;
                };
                for (member, leaf) in members {
                    push(group, member, leaf, errors);
                }
            }
        }
    }
    out
}

/// The counterpart symbols in a leaf's `field`: absent means none, but a
/// value that is not an array, or an element that is not a string, is an
/// error, so a typo cannot hide behind a valid sibling symbol.
fn symbol_list(
    value: Option<&Value>,
    at: &str,
    field: &str,
    errors: &mut Vec<String>,
) -> Vec<String> {
    let Some(value) = value else {
        return Vec::new();
    };
    let Some(items) = value.as_array() else {
        errors.push(format!(
            "{at}: `{field}` must be an array of strings, found {}",
            describe(value)
        ));
        return Vec::new();
    };
    items
        .iter()
        .enumerate()
        .filter_map(|(i, item)| {
            let symbol = item.as_str();
            if symbol.is_none() {
                errors.push(format!(
                    "{at}: `{field}[{i}]` must be a string, found {}",
                    describe(item)
                ));
            }
            symbol.map(str::to_string)
        })
        .collect()
}

/// How a JSON value reads in an "expected …, found …" diagnostic.
fn describe(value: &Value) -> &'static str {
    match value {
        Value::Null => "nothing",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

/// `value` as a JSON object, or an error naming `at`.
fn object<'a>(
    value: &'a Value,
    at: &str,
    errors: &mut Vec<String>,
) -> Option<&'a Map<String, Value>> {
    let map = value.as_object();
    if map.is_none() {
        errors.push(format!(
            "{at}: expected an object, found {}",
            describe(value)
        ));
    }
    map
}

fn load(root: &Path, errors: &mut Vec<String>) -> Result<Inputs, String> {
    let ledger = read_json(root, LEDGER)?;
    let go_tree = read_json(root, GO_MAPPING)?;
    let inventory = read_json(root, INVENTORY)?;
    let rust = flatten(&ledger, LEDGER, "rust", errors);
    let go = flatten(&go_tree, GO_MAPPING, "go_symbols", errors);
    let inventory_only = inventory_only_types(&inventory, &rust, &go, errors);
    Ok(Inputs {
        ledger,
        rust,
        go,
        inventory,
        inventory_only,
    })
}

// --- .NET declarations neither catalog lists ------------------------------

/// `Microsoft.Agents.AI.AIContextProvider+InvokedContext` →
/// `("Microsoft.Agents.AI", "AIContextProvider.InvokedContext")`, and a
/// generic arity becomes the parameter names (``AgentResponse`1`` →
/// `AgentResponse<T>`): the inverse of [`clr_type_key`]. A nested type's
/// `generic_parameters` list the enclosing types' parameters first.
fn catalog_type_key(clr: &str, generic_parameters: &Value) -> (String, String) {
    let (outer, nested) = match clr.split_once('+') {
        Some((outer, nested)) => (outer, Some(nested)),
        None => (clr, None),
    };
    let (ns, first) = outer.rsplit_once('.').unwrap_or(("", outer));
    let names: Vec<&str> = generic_parameters
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|p| p["name"].as_str())
        .collect();
    let mut next = 0;
    let segments: Vec<String> = std::iter::once(first)
        .chain(nested.into_iter().flat_map(|n| n.split('+')))
        .map(|segment| match segment.split_once('`') {
            Some((name, arity)) => {
                let arity: usize = arity.parse().unwrap_or(0);
                let params: Vec<String> = (next..next + arity)
                    .map(|i| {
                        names
                            .get(i)
                            .map_or_else(|| format!("T{i}"), |n| n.to_string())
                    })
                    .collect();
                next += arity;
                format!("{name}<{}>", params.join(", "))
            }
            None => segment.to_string(),
        })
        .collect();
    (ns.to_string(), segments.join("."))
}

/// Types in the .NET inventory that neither Go's catalog nor the Rust ledger
/// lists at all, as type-level keys. They are unreviewed: a refresh that adds
/// a .NET type surfaces it here even before Go catalogs it. Members are not
/// listed one by one, because the inventory also carries compiler-generated
/// members (record equality, `<Clone>$`, `Deconstruct`) that no catalog
/// reviews; a type is counted once and its members follow once it is
/// reviewed.
fn inventory_only_types(
    inventory: &Value,
    rust: &BTreeMap<Key, Leaf>,
    go: &BTreeMap<Key, Leaf>,
    errors: &mut Vec<String>,
) -> BTreeSet<Key> {
    let Some(types) = object(
        &inventory["types"],
        &format!("{INVENTORY}: `types`"),
        errors,
    ) else {
        return BTreeSet::new();
    };
    let catalogued: BTreeSet<String> = go
        .keys()
        .chain(rust.keys())
        .map(|(ns, ty, ..)| clr_type_key(ns, ty))
        .collect();
    types
        .iter()
        .filter(|(clr, _)| !catalogued.contains(*clr))
        .map(|(clr, entry)| {
            let (ns, ty) = catalog_type_key(clr, &entry["generic_parameters"]);
            (ns, ty, "type".to_string(), String::new())
        })
        .collect()
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
            Some((name, args)) => format!(
                "{name}`{}",
                split_top_level(args.trim_end_matches('>')).len()
            ),
            None => s,
        })
        .collect();
    format!("{ns}.{}", segments.join("+"))
}

/// Comma-separated entries at nesting depth zero.
fn split_top_level(list: &str) -> Vec<&str> {
    if list.trim().is_empty() {
        return Vec::new();
    }
    let mut depth = 0i32;
    let mut start = 0;
    let mut out = Vec::new();
    for (i, c) in list.char_indices() {
        match c {
            '<' | '(' | '[' => depth += 1,
            '>' | ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                out.push(&list[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(&list[start..]);
    out
}

/// A parameter's type in one spelling, as tokens: the ledger's C# source
/// (`Func<JsonElement?, string>`, `ref int`) and the inventory's CLR
/// signature (``System.Func`2<System.Nullable`1<System.Text.Json.JsonElement>,System.String>``,
/// `System.Int32&`) both become `Func < JsonElement , String >` / `Int32`.
/// Names lose their namespace, enclosing type and generic arity; C# keyword
/// aliases take their CLR names; nullability (`?`, `Nullable<T>`), by-ref
/// (`&`) and parameter modifiers drop out; and a generic placeholder (`!0`,
/// `!!0`) becomes `*`, which matches any one name.
fn parameter_type(spelling: &str) -> Vec<String> {
    let mut rest = spelling.trim();
    while let Some((word, tail)) = rest.split_once(char::is_whitespace) {
        if !matches!(word, "this" | "ref" | "out" | "in" | "params" | "scoped") {
            break;
        }
        rest = tail.trim_start();
    }
    let mut tokens = Vec::new();
    let mut chars = rest.chars().peekable();
    while let Some(c) = chars.next() {
        if c.is_alphanumeric() || c == '_' {
            let mut name = c.to_string();
            while let Some(&n) = chars.peek() {
                if !(n.is_alphanumeric() || matches!(n, '_' | '.' | '+' | '`')) {
                    break;
                }
                name.push(n);
                chars.next();
            }
            let name = name.rsplit(['.', '+']).next().unwrap_or_default();
            let name = name.split('`').next().unwrap_or_default();
            tokens.push(clr_alias(name).to_string());
        } else if c == '!' {
            while chars
                .peek()
                .is_some_and(|n| *n == '!' || n.is_ascii_digit())
            {
                chars.next();
            }
            tokens.push("*".to_string());
        } else if !(c.is_whitespace() || matches!(c, '?' | '&')) {
            tokens.push(c.to_string());
        }
    }
    // `Nullable<T>` → `T`: drop the name and its brackets, keep the argument.
    let mut out = Vec::new();
    let mut open = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        if tokens[i] == "Nullable" && tokens.get(i + 1).is_some_and(|t| t == "<") {
            open.push(false);
            i += 2;
            continue;
        }
        if tokens[i] == "<" {
            open.push(true);
        } else if tokens[i] == ">" && open.pop() == Some(false) {
            i += 1;
            continue;
        }
        out.push(std::mem::take(&mut tokens[i]));
        i += 1;
    }
    out
}

/// The CLR name of a C# keyword type, or `name` unchanged.
fn clr_alias(name: &str) -> &str {
    match name {
        "string" => "String",
        "bool" => "Boolean",
        "int" => "Int32",
        "long" => "Int64",
        "short" => "Int16",
        "byte" => "Byte",
        "sbyte" => "SByte",
        "uint" => "UInt32",
        "ulong" => "UInt64",
        "ushort" => "UInt16",
        "float" => "Single",
        "double" => "Double",
        "decimal" => "Decimal",
        "char" => "Char",
        "object" => "Object",
        "void" => "Void",
        "nint" => "IntPtr",
        "nuint" => "UIntPtr",
        other => other,
    }
}

/// `(base name, parameter types or None for a property-like key)`, the
/// types normalised by [`parameter_type`].
fn member_shape(key: &str) -> (String, Option<Vec<Vec<String>>>) {
    let head = key.split(" -> ").next().unwrap_or(key);
    let (name, params) = match head.split_once('(') {
        Some((name, rest)) => {
            let list = rest.strip_suffix(')').unwrap_or(rest);
            let params = split_top_level(list)
                .into_iter()
                .map(parameter_type)
                .collect();
            (name, Some(params))
        }
        None => (head, None),
    };
    let name = name.split(['<', '`']).next().unwrap_or(name).trim();
    (name.to_string(), params)
}

/// Whether two normalised parameter lists name the same overload: same
/// length, and each type equal token for token, `*` matching any one token.
fn same_parameters(a: &[Vec<String>], b: &[Vec<String>]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.len() == y.len() && x.iter().zip(y).all(|(s, t)| s == t || s == "*" || t == "*")
        })
}

fn resolves_in_inventory(inventory: &Value, key: &Key) -> bool {
    let (ns, ty, group, member) = key;
    let Some(entry) = inventory["types"].get(clr_type_key(ns, ty)) else {
        return false;
    };
    if group == "type" {
        return true;
    }
    let (mut name, params) = member_shape(member);
    if group == "constructors" {
        name = ".ctor".to_string();
    }
    entry[group.as_str()].as_object().is_some_and(|members| {
        members.keys().any(|k| {
            let (n, p) = member_shape(k);
            n == name
                && match (&params, &p) {
                    (Some(a), Some(b)) => same_parameters(a, b),
                    _ => true,
                }
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
    // Go's statuses feed the report's counts, which tally only the known
    // ones, so an unsupported status would silently drop a declaration.
    for (key, leaf) in &inputs.go {
        if !STATUSES.contains(&leaf.status.as_str()) {
            errors.push(format!(
                "{GO_MAPPING}: {}: unsupported status `{}`",
                label(key),
                leaf.status
            ));
        }
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
    for key in &inputs.inventory_only {
        row(&mut rows, &key.0).unreviewed += 1;
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

/// Loads the inputs for a read-only command, failing on any parse problem so
/// a malformed ledger or catalog is never reported, or written, as partial.
fn load_strict(root: &Path) -> Result<Inputs, String> {
    let mut errors = Vec::new();
    let inputs = load(root, &mut errors)?;
    if errors.is_empty() {
        return Ok(inputs);
    }
    for e in &errors {
        eprintln!("  {e}");
    }
    Err(format!(
        "{} problem(s) reading the parity ledger; run `cargo xtask parity check`",
        errors.len()
    ))
}

pub fn summary(root: &Path) -> Result<(), String> {
    let inputs = load_strict(root)?;
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
    keys.extend(inputs.inventory_only.iter());
    for key in keys {
        let go = inputs.go.get(key).map(|l| l.status.as_str()).unwrap_or("-");
        let (rust, note) = match inputs.rust.get(key) {
            Some(l) => (l.status.as_str(), l.note.as_str()),
            None => ("unreviewed", ""),
        };
        // Every unreviewed declaration is listed, whatever Go's status: a
        // refresh that adds one must surface it for assessment.
        if matches!(rust, "unmapped" | "partial" | "unreviewed") {
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
    let inputs = load_strict(root)?;
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
    let inputs = load_strict(root)?;
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
    let (dotnet_only, in_go): (Vec<_>, Vec<_>) = unreviewed
        .into_iter()
        .partition(|(key, ..)| inputs.inventory_only.contains(key));
    let _ = writeln!(
        out,
        "{total_unreviewed} declarations the Rust ledger has not assessed yet.\n"
    );
    let _ = writeln!(
        out,
        "### In Go's catalog\n\n{} declarations, by type:\n",
        in_go.len()
    );
    let mut by_type: BTreeMap<String, usize> = BTreeMap::new();
    for (key, ..) in &in_go {
        *by_type.entry(format!("{}::{}", key.0, key.1)).or_default() += 1;
    }
    if by_type.is_empty() {
        let _ = writeln!(out, "None.");
    }
    for (ty, n) in by_type {
        let _ = writeln!(out, "- `{ty}` ({n})");
    }
    let _ = writeln!(
        out,
        "\n### Only in the .NET inventory\n\n{} types that neither Go's catalog nor the Rust ledger lists. \
         Each counts once; review its members when it is assessed.\n",
        dotnet_only.len()
    );
    if dotnet_only.is_empty() {
        let _ = writeln!(out, "None.");
    }
    for (key, ..) in &dotnet_only {
        let _ = writeln!(out, "- `{}::{}`", key.0, key.1);
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
    fn report_refuses_to_write_from_a_malformed_ledger() {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let root = std::env::temp_dir().join(format!("parity-report-{}", std::process::id()));
        let upstream = root.join("docs/parity/upstream");
        fs::create_dir_all(&upstream).unwrap();
        for file in [INVENTORY, GO_MAPPING] {
            fs::copy(repo.join(file), root.join(file)).unwrap();
        }
        let ledger = serde_json::json!({
            "schema_version": 1,
            "namespaces": { "Microsoft.Agents.AI": { "AIAgent": { "bogus_group": {} } } }
        });
        fs::write(root.join(LEDGER), ledger.to_string()).unwrap();

        let result = write_report(&root);
        let written = root.join(REPORT).exists();
        fs::remove_dir_all(&root).unwrap();
        assert!(result.is_err(), "report accepted a malformed ledger");
        assert!(!written, "report was written from a malformed ledger");
    }

    fn flatten_errors(tree: Value) -> Vec<String> {
        let mut errors = Vec::new();
        let leaves = flatten(&tree, "ledger.json", "rust", &mut errors);
        assert!(leaves.is_empty(), "malformed input yielded leaves");
        errors
    }

    #[test]
    fn flatten_rejects_malformed_containers_at_every_level() {
        use serde_json::json;
        let cases = [
            (
                json!({}),
                "ledger.json: `namespaces`: expected an object, found nothing",
            ),
            (
                json!({"namespaces": []}),
                "ledger.json: `namespaces`: expected an object, found an array",
            ),
            (
                json!({"namespaces": {"N": []}}),
                "ledger.json: namespace N: expected an object, found an array",
            ),
            (
                json!({"namespaces": {"N": {"T": "x"}}}),
                "ledger.json: N::T: expected an object, found a string",
            ),
            (
                json!({"namespaces": {"N": {"T": {"methods": []}}}}),
                "ledger.json: N::T methods: expected an object, found an array",
            ),
            (
                json!({"namespaces": {"N": {"T": {"methods": {"M()": []}}}}}),
                "ledger.json: N::T methods::M(): expected an object, found an array",
            ),
            (
                json!({"namespaces": {"N": {"T": {"mapping": 1}}}}),
                "ledger.json: N::T: expected an object, found a number",
            ),
            (
                json!({"namespaces": {"N": {"T": {"method": {}}}}}),
                "ledger.json: N::T: unknown group `method`",
            ),
        ];
        for (tree, expected) in cases {
            assert_eq!(flatten_errors(tree), [expected]);
        }
    }

    #[test]
    fn flatten_reads_a_well_formed_tree() {
        let tree = serde_json::json!({"namespaces": {"N": {"T": {
            "area": "core",
            "mapping": {"status": "unmapped", "note": "n"},
            "methods": {"M()": {"status": "mapped", "note": "n", "rust": ["k::m"]}}
        }}}});
        let mut errors = Vec::new();
        let leaves = flatten(&tree, "ledger.json", "rust", &mut errors);
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(leaves.len(), 2);
        let m = &leaves[&("N".into(), "T".into(), "methods".into(), "M()".into())];
        assert_eq!(m.rust, ["k::m"]);
        assert_eq!(m.area, "core");
    }

    #[test]
    fn catalog_type_keys_invert_the_clr_spelling() {
        let params = serde_json::json!([{"name": "TInput"}, {"name": "TOutput"}]);
        assert_eq!(
            catalog_type_key("Microsoft.Agents.AI.Workflows.Executor`2", &params),
            (
                "Microsoft.Agents.AI.Workflows".into(),
                "Executor<TInput, TOutput>".into()
            )
        );
        assert_eq!(
            catalog_type_key(
                "Microsoft.Agents.AI.ChatHistoryMemoryProvider+State",
                &Value::Null
            ),
            (
                "Microsoft.Agents.AI".into(),
                "ChatHistoryMemoryProvider.State".into()
            )
        );
        // Every vendored inventory type round-trips through both spellings.
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let inventory = read_json(&root, INVENTORY).unwrap();
        for (clr, entry) in inventory["types"].as_object().unwrap() {
            let (ns, ty) = catalog_type_key(clr, &entry["generic_parameters"]);
            assert_eq!(&clr_type_key(&ns, &ty), clr);
        }
    }

    #[test]
    fn inventory_only_types_are_unreviewed_and_counted_once() {
        let inventory = serde_json::json!({"types": {
            "N.Listed": {"methods": {"M() -> System.Void": {}}},
            "N.Store": {"methods": {"A() -> System.Void": {}, "B() -> System.Void": {}}},
            "N.Box`1": {"generic_parameters": [{"name": "T"}]}
        }});
        let leaf = Leaf {
            status: "mapped".into(),
            note: "n".into(),
            rust: vec![],
            area: String::new(),
        };
        let go = BTreeMap::from([(
            (
                "N".to_string(),
                "Listed".to_string(),
                "methods".to_string(),
                "M()".to_string(),
            ),
            leaf,
        )]);
        let mut errors = Vec::new();
        let only = inventory_only_types(&inventory, &BTreeMap::new(), &go, &mut errors);
        assert!(errors.is_empty());
        let inputs = Inputs {
            ledger: Value::Null,
            rust: BTreeMap::new(),
            go,
            inventory,
            inventory_only: only,
        };
        let rows = tally(&inputs);
        // `Listed`'s method is unreviewed through Go's catalog; `Store` and
        // `Box<T>` once each through the inventory.
        assert_eq!(rows["N"].unreviewed, 3);
        let gaps: Vec<String> = gap_rows(&inputs).iter().map(|(k, ..)| label(k)).collect();
        assert_eq!(gaps, ["N::Box<T>", "N::Listed methods::M()", "N::Store"]);
        let report = render_report(&inputs);
        assert!(report.contains("- `N::Store`"), "{report}");
    }

    #[test]
    fn go_leaves_with_unsupported_statuses_are_rejected() {
        let tree = serde_json::json!({"namespaces": {"N": {"T": {
            "methods": {"M()": {"status": "done", "note": "n"}}
        }}}});
        let mut errors = Vec::new();
        let go = flatten(&tree, GO_MAPPING, "go_symbols", &mut errors);
        let inputs = Inputs {
            ledger: serde_json::json!({"schema_version": 1}),
            rust: BTreeMap::new(),
            go,
            inventory: Value::Null,
            inventory_only: BTreeSet::new(),
        };
        validate(&inputs, &RustIndex::new(), &mut errors);
        assert_eq!(
            errors,
            [format!(
                "{GO_MAPPING}: N::T methods::M(): unsupported status `done`"
            )]
        );
    }

    #[test]
    fn flatten_rejects_non_string_symbols() {
        use serde_json::json;
        let leaf = |rust: Value| {
            json!({"namespaces": {"N": {"T": {"methods": {"M()": {
                "status": "mapped", "note": "n", "rust": rust
            }}}}}})
        };
        let at = "ledger.json: N::T methods::M()";
        let cases = [
            (
                json!(["k::m", null, 3]),
                vec![
                    format!("{at}: `rust[1]` must be a string, found nothing"),
                    format!("{at}: `rust[2]` must be a string, found a number"),
                ],
            ),
            (
                json!("k::m"),
                vec![format!(
                    "{at}: `rust` must be an array of strings, found a string"
                )],
            ),
            (
                Value::Null,
                vec![format!(
                    "{at}: `rust` must be an array of strings, found nothing"
                )],
            ),
        ];
        for (rust, expected) in cases {
            let mut errors = Vec::new();
            flatten(&leaf(rust), "ledger.json", "rust", &mut errors);
            assert_eq!(errors, expected);
        }
    }

    fn shape(key: &str) -> (String, Option<Vec<String>>) {
        let (name, params) = member_shape(key);
        let params = params.map(|ps| ps.into_iter().map(|p| p.join(" ")).collect());
        (name, params)
    }

    #[test]
    fn member_shapes_ignore_generics_and_return_types() {
        assert_eq!(
            shape("RunAsync<T>(string, AgentSession, CancellationToken)"),
            (
                "RunAsync".into(),
                Some(vec![
                    "String".into(),
                    "AgentSession".into(),
                    "CancellationToken".into()
                ])
            )
        );
        assert_eq!(
            shape("GetService``1(System.Object) -> !!0"),
            ("GetService".into(), Some(vec!["Object".into()]))
        );
        assert_eq!(
            shape("AddEdge(Microsoft.Agents.AI.Workflows.ExecutorBinding,System.Func`2<System.Object,System.Boolean>) -> X"),
            (
                "AddEdge".into(),
                Some(vec![
                    "ExecutorBinding".into(),
                    "Func < Object , Boolean >".into()
                ])
            )
        );
        assert_eq!(shape("Name -> System.String"), ("Name".into(), None));
        assert_eq!(shape("Build()"), ("Build".into(), Some(vec![])));
    }

    #[test]
    fn parameter_types_agree_across_spellings() {
        let pairs = [
            ("string?", "System.String"),
            ("int?", "System.Nullable`1<System.Int32>"),
            (
                "Func<JsonElement?, AIFunctionArguments>",
                "System.Func`2<System.Nullable`1<System.Text.Json.JsonElement>,Microsoft.Extensions.AI.AIFunctionArguments>",
            ),
            (
                "AIContextProvider.InvokedContext",
                "Microsoft.Agents.AI.AIContextProvider+InvokedContext",
            ),
            ("ref int", "System.Int32&"),
            ("params ChatMessage[]", "Microsoft.Extensions.AI.ChatMessage[]"),
        ];
        for (ledger, inventory) in pairs {
            assert_eq!(
                parameter_type(ledger),
                parameter_type(inventory),
                "{ledger}"
            );
        }
        assert_eq!(parameter_type("!!0"), ["*"]);
    }

    #[test]
    fn inventory_lookup_compares_parameter_types() {
        let inventory = serde_json::json!({"types": {"N.T": {
            "methods": {
                "Foo(System.Int32) -> System.Void": {},
                "Bar``1(!!0,System.String) -> System.Void": {}
            },
            "constructors": {".ctor(System.Nullable`1<System.Int32>)": {}},
            "properties": {"Name -> System.String": {}}
        }}});
        let key = |group: &str, member: &str| {
            (
                "N".to_string(),
                "T".to_string(),
                group.to_string(),
                member.to_string(),
            )
        };
        for (group, member, found) in [
            ("methods", "Foo(int)", true),
            ("methods", "Foo(string)", false),
            ("methods", "Foo(int, int)", false),
            ("methods", "Bar<TItem>(TItem, string)", true),
            ("methods", "Bar<TItem>(TItem, int)", false),
            ("constructors", "T(int?)", true),
            ("constructors", "T(long?)", false),
            ("properties", "Name", true),
        ] {
            assert_eq!(
                resolves_in_inventory(&inventory, &key(group, member)),
                found,
                "{group}::{member}"
            );
        }
    }

    #[test]
    fn every_go_catalog_member_resolves_by_parameter_types() {
        // Go's catalog spells keys the way the ledger does, so each of its
        // members on an inventory type must find its overload by type.
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let mut errors = Vec::new();
        let tree = read_json(&root, GO_MAPPING).unwrap();
        let go = flatten(&tree, GO_MAPPING, "go_symbols", &mut errors);
        assert!(errors.is_empty(), "{errors:?}");
        let inventory = read_json(&root, INVENTORY).unwrap();
        let unresolved: Vec<String> = go
            .keys()
            .filter(|(ns, ty, group, _)| {
                group != "type" && inventory["types"].get(clr_type_key(ns, ty)).is_some()
            })
            .filter(|key| !resolves_in_inventory(&inventory, key))
            .map(label)
            .collect();
        assert!(unresolved.is_empty(), "{unresolved:#?}");
    }
}
