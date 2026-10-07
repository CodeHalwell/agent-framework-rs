//! `cargo xtask features check`: enforces the feature-stage rules in
//! `docs/feature-stages.md` against every crate's `Cargo.toml`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

const DOC: &str = "docs/feature-stages.md";
const UMBRELLA: &str = "agent-framework";
const PREFIX: &str = "experimental-";

/// One crate's manifest, as far as the feature-stage rules care.
#[derive(Debug, Default, Clone)]
struct Crate {
    /// `feature name -> what it enables`.
    features: BTreeMap<String, Vec<String>>,
    /// `dependency key -> package name` (they differ only when renamed).
    deps: BTreeMap<String, String>,
    /// `(dependency key, feature)` for every `features = [...]` entry on a
    /// dependency, which turns that feature on in every build.
    dep_features: Vec<(String, String)>,
}

/// `crate name -> manifest`.
type Crates = BTreeMap<String, Crate>;

/// `feature -> (upstream id, crates)`, from the doc table.
type Table = BTreeMap<String, (String, BTreeSet<String>)>;

fn parse_manifest(text: &str) -> Result<(String, Crate), String> {
    let doc: toml::Table = toml::from_str(text).map_err(|e| e.to_string())?;
    let name = doc
        .get("package")
        .and_then(|p| p.get("name"))
        .and_then(|n| n.as_str())
        .ok_or("no package name")?
        .to_string();
    let mut krate = Crate::default();
    if let Some(table) = doc.get("features").and_then(|f| f.as_table()) {
        for (feature, enables) in table {
            let list = enables
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            krate.features.insert(feature.clone(), list);
        }
    }
    // Dev-dependencies never reach a downstream build, so only the normal
    // and build dependency tables count.
    for section in ["dependencies", "build-dependencies"] {
        let Some(table) = doc.get(section).and_then(|d| d.as_table()) else {
            continue;
        };
        for (key, spec) in table {
            let package = spec
                .get("package")
                .and_then(|p| p.as_str())
                .unwrap_or(key)
                .to_string();
            krate.deps.insert(key.clone(), package);
            if let Some(list) = spec.get("features").and_then(|f| f.as_array()) {
                for f in list.iter().filter_map(|v| v.as_str()) {
                    krate.dep_features.push((key.clone(), f.to_string()));
                }
            }
        }
    }
    Ok((name, krate))
}

fn read_crates(root: &Path) -> Result<Crates, String> {
    let mut out = Crates::new();
    let dir = root.join("crates");
    for entry in fs::read_dir(&dir).map_err(|e| format!("reading {}: {e}", dir.display()))? {
        let manifest = entry.map_err(|e| e.to_string())?.path().join("Cargo.toml");
        let Ok(text) = fs::read_to_string(&manifest) else {
            continue;
        };
        let (name, krate) =
            parse_manifest(&text).map_err(|e| format!("{}: {e}", manifest.display()))?;
        out.insert(name, krate);
    }
    Ok(out)
}

/// Every `(crate, feature)` that turning on `feature` of `krate` turns on,
/// following local feature aliases and `dep/feature` / `dep?/feature`
/// strings into other workspace crates. Includes the starting pair.
fn closure(all: &Crates, krate: &str, feature: &str) -> BTreeSet<(String, String)> {
    let mut seen = BTreeSet::new();
    let mut stack = vec![(krate.to_string(), feature.to_string())];
    while let Some((c, f)) = stack.pop() {
        if !seen.insert((c.clone(), f.clone())) {
            continue;
        }
        let Some(manifest) = all.get(&c) else {
            continue;
        };
        for entry in manifest.features.get(&f).into_iter().flatten() {
            if entry.starts_with("dep:") {
                continue;
            }
            match entry.split_once('/') {
                Some((dep, dep_feature)) => {
                    let dep = dep.trim_end_matches('?');
                    let package = manifest.deps.get(dep).map_or(dep, String::as_str);
                    stack.push((package.to_string(), dep_feature.to_string()));
                }
                // A bare name is a local feature (or an optional dependency's
                // implicit feature, which `features` does not list).
                None => stack.push((c.clone(), entry.clone())),
            }
        }
    }
    seen
}

/// Rows of the table between the `feature-table` markers:
/// `feature -> (upstream id, crates)`.
fn read_table(root: &Path) -> Result<Table, String> {
    let text = fs::read_to_string(root.join(DOC)).map_err(|e| format!("reading {DOC}: {e}"))?;
    let start = text
        .find("<!-- feature-table")
        .ok_or_else(|| format!("{DOC}: missing the feature-table marker"))?;
    let end = text
        .find("<!-- /feature-table -->")
        .ok_or_else(|| format!("{DOC}: missing the closing feature-table marker"))?;
    let mut rows = BTreeMap::new();
    for line in text[start..end].lines().filter(|l| l.starts_with("| `")) {
        let cells: Vec<&str> = line.split('|').map(str::trim).collect();
        // ["", feature, id, crates, what, ""]
        if cells.len() < 5 {
            return Err(format!("{DOC}: malformed row `{line}`"));
        }
        let strip = |s: &str| s.trim_matches('`').to_string();
        let crates = cells[3]
            .split(',')
            .map(|c| strip(c.trim()))
            .filter(|c| !c.is_empty())
            .collect();
        rows.insert(strip(cells[1]), (strip(cells[2]), crates));
    }
    Ok(rows)
}

pub fn check(root: &Path) -> Result<(), String> {
    let all = read_crates(root)?;
    let table = read_table(root)?;
    let (errors, count) = problems(&all, &table);
    if errors.is_empty() {
        println!("feature stages OK: {count} experimental feature(s)");
        Ok(())
    } else {
        for e in &errors {
            eprintln!("  {e}");
        }
        Err(format!("{} feature-stage problem(s)", errors.len()))
    }
}

/// Every rule violation, and how many experimental features crates declare.
fn problems(all: &Crates, table: &Table) -> (Vec<String>, usize) {
    let mut errors = Vec::new();

    // Which non-umbrella crates declare each experimental feature.
    let mut declared: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (krate, manifest) in all {
        for feature in manifest.features.keys() {
            if feature.starts_with(PREFIX) && krate != UMBRELLA {
                declared
                    .entry(feature.clone())
                    .or_default()
                    .insert(krate.clone());
            }
        }
    }

    // `default` (any crate) and the umbrella's `full` must not reach an
    // experimental feature, directly or through aliases and dependencies.
    for (krate, manifest) in all {
        let mut roots = vec!["default"];
        if krate == UMBRELLA {
            roots.push("full");
        }
        for root_feature in roots {
            if !manifest.features.contains_key(root_feature) {
                continue;
            }
            for (c, f) in closure(all, krate, root_feature) {
                if f.starts_with(PREFIX) {
                    let target = if &c == krate { f } else { format!("{c}/{f}") };
                    errors.push(format!(
                        "{krate}: `{root_feature}` must not enable `{target}`"
                    ));
                }
            }
        }
        // A dependency's `features = [...]` is on in every build.
        for (dep, f) in &manifest.dep_features {
            if f.starts_with(PREFIX) {
                errors.push(format!(
                    "{krate}: dependency `{dep}` must not turn on `{f}` unconditionally"
                ));
            }
        }
    }

    // Same-name forwarding: a crate declaring an experimental feature must
    // forward it to every dependency that declares the same feature.
    for (krate, manifest) in all {
        for (feature, enables) in manifest
            .features
            .iter()
            .filter(|(f, _)| f.starts_with(PREFIX))
        {
            for (dep, package) in &manifest.deps {
                let declares = all
                    .get(package)
                    .is_some_and(|d| d.features.contains_key(feature));
                if !declares {
                    continue;
                }
                let forwarded = enables
                    .iter()
                    .any(|e| e == &format!("{dep}/{feature}") || e == &format!("{dep}?/{feature}"));
                if !forwarded {
                    errors.push(format!(
                        "{krate}'s `{feature}` does not enable `{dep}/{feature}`"
                    ));
                }
            }
        }
    }

    let umbrella = all.get(UMBRELLA).map(|c| &c.features);
    let umbrella_all: BTreeSet<&str> = umbrella
        .and_then(|u| u.get("experimental"))
        .map(|v| v.iter().map(String::as_str).collect())
        .unwrap_or_default();

    for (feature, crates) in &declared {
        match table.get(feature) {
            None => errors.push(format!("`{feature}` is not in the {DOC} table")),
            Some((id, listed)) => {
                let expected = format!("{PREFIX}{}", id.to_lowercase().replace('_', "-"));
                if &expected != feature {
                    errors.push(format!(
                        "`{feature}` should be named `{expected}` after `{id}`"
                    ));
                }
                if listed != crates {
                    errors.push(format!(
                        "{DOC}: `{feature}` lists crates {listed:?}, but it is declared by {crates:?}"
                    ));
                }
            }
        }
        if !umbrella.is_some_and(|u| u.contains_key(feature)) {
            errors.push(format!("{UMBRELLA} does not re-expose `{feature}`"));
        }
        if !umbrella_all.contains(feature.as_str()) {
            errors.push(format!(
                "{UMBRELLA}'s `experimental` does not include `{feature}`"
            ));
        }
    }
    for feature in table.keys() {
        if !declared.contains_key(feature) {
            errors.push(format!("{DOC} lists `{feature}`, which no crate declares"));
        }
    }
    for feature in umbrella
        .into_iter()
        .flat_map(|u| u.keys())
        .filter(|f| f.starts_with(PREFIX))
    {
        if !declared.contains_key(feature) {
            errors.push(format!(
                "{UMBRELLA} declares `{feature}`, which no crate declares"
            ));
        }
    }
    (errors, declared.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    const CORE: &str = r#"
        [package]
        name = "agent-framework-core"
        [features]
        experimental-vector-stores = []
    "#;

    const COSMOS: &str = r#"
        [package]
        name = "agent-framework-cosmos"
        [features]
        experimental-vector-stores = ["agent-framework-core/experimental-vector-stores"]
        [dependencies]
        agent-framework-core = { workspace = true }
    "#;

    const UMBRELLA_OK: &str = r#"
        [package]
        name = "agent-framework"
        [features]
        default = ["openai"]
        openai = []
        cosmos = ["dep:agent-framework-cosmos"]
        experimental-vector-stores = [
            "agent-framework-core/experimental-vector-stores",
            "agent-framework-cosmos?/experimental-vector-stores",
        ]
        experimental = ["experimental-vector-stores"]
        full = ["openai", "cosmos"]
        [dependencies]
        agent-framework-core = { workspace = true }
        agent-framework-cosmos = { workspace = true, optional = true }
    "#;

    fn crates(manifests: &[&str]) -> Crates {
        manifests
            .iter()
            .map(|m| parse_manifest(m).unwrap())
            .collect()
    }

    fn table() -> Table {
        let crates = ["agent-framework-core", "agent-framework-cosmos"]
            .into_iter()
            .map(String::from)
            .collect();
        BTreeMap::from([(
            "experimental-vector-stores".to_string(),
            ("VECTOR_STORES".to_string(), crates),
        )])
    }

    fn errors(manifests: &[&str]) -> Vec<String> {
        problems(&crates(manifests), &table()).0
    }

    #[test]
    fn a_consistent_workspace_passes() {
        assert_eq!(errors(&[CORE, COSMOS, UMBRELLA_OK]), Vec::<String>::new());
    }

    #[test]
    fn full_reaching_experimental_through_a_local_alias_is_caught() {
        let umbrella = UMBRELLA_OK.replace(
            r#"full = ["openai", "cosmos"]"#,
            r#"stores = ["experimental-vector-stores"]
        full = ["openai", "cosmos", "stores"]"#,
        );
        let errs = errors(&[CORE, COSMOS, &umbrella]);
        assert!(
            errs.iter().any(
                |e| e == "agent-framework: `full` must not enable `experimental-vector-stores`"
            ),
            "{errs:?}"
        );
    }

    #[test]
    fn default_reaching_experimental_through_a_dependency_feature_is_caught() {
        for spelling in ["agent-framework-cosmos", "agent-framework-cosmos?"] {
            let umbrella = UMBRELLA_OK.replace(
                r#"default = ["openai"]"#,
                &format!(r#"default = ["openai", "{spelling}/experimental-vector-stores"]"#),
            );
            let errs = errors(&[CORE, COSMOS, &umbrella]);
            assert!(
                errs.iter().any(|e| e
                    == "agent-framework: `default` must not enable `agent-framework-cosmos/experimental-vector-stores`"),
                "{spelling}: {errs:?}"
            );
        }
    }

    #[test]
    fn default_reaching_experimental_through_an_alias_chain_into_a_dependency_is_caught() {
        let cosmos = COSMOS.replace(
            "[dependencies]",
            r#"default = ["vec"]
        vec = ["agent-framework-core/experimental-vector-stores"]
        [dependencies]"#,
        );
        let errs = errors(&[CORE, &cosmos, UMBRELLA_OK]);
        assert!(
            errs.iter().any(|e| e
                == "agent-framework-cosmos: `default` must not enable `agent-framework-core/experimental-vector-stores`"),
            "{errs:?}"
        );
    }

    #[test]
    fn a_dependency_turning_experimental_on_unconditionally_is_caught() {
        let cosmos = COSMOS.replace(
            "agent-framework-core = { workspace = true }",
            r#"agent-framework-core = { workspace = true, features = ["experimental-vector-stores"] }"#,
        );
        let errs = errors(&[CORE, &cosmos, UMBRELLA_OK]);
        assert!(
            errs.iter().any(|e| e.starts_with("agent-framework-cosmos: dependency `agent-framework-core`")),
            "{errs:?}"
        );
    }

    #[test]
    fn a_non_umbrella_crate_must_forward_its_feature_to_a_declaring_dependency() {
        let cosmos = COSMOS.replace(
            r#"experimental-vector-stores = ["agent-framework-core/experimental-vector-stores"]"#,
            "experimental-vector-stores = []",
        );
        let errs = errors(&[CORE, &cosmos, UMBRELLA_OK]);
        assert_eq!(
            errs,
            vec![
                "agent-framework-cosmos's `experimental-vector-stores` does not enable `agent-framework-core/experimental-vector-stores`"
            ]
        );
    }

    #[test]
    fn the_umbrella_must_forward_to_every_declaring_dependency() {
        let umbrella = UMBRELLA_OK.replace(
            r#""agent-framework-cosmos?/experimental-vector-stores","#,
            "",
        );
        let errs = errors(&[CORE, COSMOS, &umbrella]);
        assert_eq!(
            errs,
            vec![
                "agent-framework's `experimental-vector-stores` does not enable `agent-framework-cosmos/experimental-vector-stores`"
            ]
        );
    }
}
