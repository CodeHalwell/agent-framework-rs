//! `cargo xtask features check`: enforces the feature-stage rules in
//! `docs/feature-stages.md` against every crate's `Cargo.toml`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

const DOC: &str = "docs/feature-stages.md";
const UMBRELLA: &str = "agent-framework";
const PREFIX: &str = "experimental-";

/// `crate name -> (feature name -> what it enables)`.
type Features = BTreeMap<String, BTreeMap<String, Vec<String>>>;

fn read_features(root: &Path) -> Result<Features, String> {
    let mut out = Features::new();
    let dir = root.join("crates");
    for entry in fs::read_dir(&dir).map_err(|e| format!("reading {}: {e}", dir.display()))? {
        let manifest = entry.map_err(|e| e.to_string())?.path().join("Cargo.toml");
        let Ok(text) = fs::read_to_string(&manifest) else {
            continue;
        };
        let doc: toml::Table =
            toml::from_str(&text).map_err(|e| format!("{}: {e}", manifest.display()))?;
        let name = doc["package"]["name"]
            .as_str()
            .ok_or_else(|| format!("{}: no package name", manifest.display()))?
            .to_string();
        let mut features = BTreeMap::new();
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
                features.insert(feature.clone(), list);
            }
        }
        out.insert(name, features);
    }
    Ok(out)
}

/// Rows of the table between the `feature-table` markers:
/// `feature -> (upstream id, crates)`.
fn read_table(root: &Path) -> Result<BTreeMap<String, (String, BTreeSet<String>)>, String> {
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
    let all = read_features(root)?;
    let table = read_table(root)?;
    let mut errors = Vec::new();

    // Which non-umbrella crates declare each experimental feature.
    let mut declared: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (krate, features) in &all {
        for (feature, enables) in features {
            if feature.starts_with(PREFIX) && krate != UMBRELLA {
                declared
                    .entry(feature.clone())
                    .or_default()
                    .insert(krate.clone());
            }
            if feature == "default" || (krate == UMBRELLA && feature == "full") {
                for f in enables.iter().filter(|f| f.starts_with(PREFIX)) {
                    errors.push(format!("{krate}: `{feature}` must not enable `{f}`"));
                }
            }
        }
    }

    let umbrella = all.get(UMBRELLA).cloned().unwrap_or_default();
    let umbrella_all: BTreeSet<&str> = umbrella
        .get("experimental")
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
        match umbrella.get(feature) {
            None => errors.push(format!("{UMBRELLA} does not re-expose `{feature}`")),
            Some(enables) => {
                for krate in crates {
                    let forwarded = enables.iter().any(|e| {
                        e == &format!("{krate}/{feature}") || e == &format!("{krate}?/{feature}")
                    });
                    if !forwarded {
                        errors.push(format!(
                            "{UMBRELLA}'s `{feature}` does not enable `{krate}/{feature}`"
                        ));
                    }
                }
            }
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
    for feature in umbrella.keys().filter(|f| f.starts_with(PREFIX)) {
        if !declared.contains_key(feature) {
            errors.push(format!(
                "{UMBRELLA} declares `{feature}`, which no crate declares"
            ));
        }
    }

    if errors.is_empty() {
        println!(
            "feature stages OK: {} experimental feature(s)",
            declared.len()
        );
        Ok(())
    } else {
        for e in &errors {
            eprintln!("  {e}");
        }
        Err(format!("{} feature-stage problem(s)", errors.len()))
    }
}
