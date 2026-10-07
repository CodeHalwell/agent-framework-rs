//! Repository tooling: `parity` keeps the symbol-level parity ledger in
//! `docs/parity/` honest, and `features` enforces the feature-stage rules in
//! `docs/feature-stages.md`.
//!
//! ```text
//! cargo xtask parity index     # rebuild the Rust public-API index
//! cargo xtask parity check     # validate the ledger (rebuilds the index)
//! cargo xtask parity summary   # counts by namespace, Rust beside Go
//! cargo xtask parity gaps      # what Go or .NET has and Rust does not
//! cargo xtask parity report    # regenerate docs/parity/STATUS.md
//! cargo xtask features check   # experimental-* features follow the rules
//! ```
//!
//! See `docs/parity/README.md` for the ledger format.

mod features;
mod index;
mod ledger;

use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let root = workspace_root();
    let result = match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["parity", "index"] => index::build(&root).map(|idx| {
            println!("indexed {} public Rust symbols", idx.len());
        }),
        ["parity", "check", rest @ ..] => ledger::check(&root, !rest.contains(&"--no-build")),
        ["parity", "summary"] => ledger::summary(&root),
        ["parity", "gaps", rest @ ..] => ledger::gaps(&root, rest.first().copied()),
        ["parity", "report"] => ledger::write_report(&root),
        ["features", "check"] => features::check(&root),
        _ => {
            eprintln!(
                "usage: cargo xtask parity <index | check [--no-build] | summary | gaps [namespace] | report>\n       cargo xtask features check"
            );
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn workspace_root() -> PathBuf {
    // xtask/ sits directly under the workspace root.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask has a parent directory")
        .to_path_buf()
}
