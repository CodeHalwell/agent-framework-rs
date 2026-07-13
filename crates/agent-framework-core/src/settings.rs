//! Lightweight settings helpers: a secret-masking string newtype and a
//! precedence-based value loader.
//!
//! This mirrors upstream's `_settings.py`, which replaced a
//! `pydantic-settings`-based `AFBaseSettings` with a function-based loader
//! (`load_settings`) plus a `repr`-masking `SecretString`. Rather than port
//! the Python `TypedDict`-driven schema loader (which leans on runtime
//! reflection that has no idiomatic Rust equivalent), this module provides
//! the two reusable primitives:
//!
//! - [`SecretString`] — a `String` newtype whose [`Debug`]/[`Display`](std::fmt::Display) impls
//!   mask the value so secrets never leak into logs, while still
//!   (de)serializing to the real value and round-tripping through
//!   `serde_json`.
//! - [`load_setting`] — a single-value loader implementing the same
//!   precedence as upstream's `load_settings`: explicit override, then a
//!   `.env` file, then the process environment, then a default.
//!
//! ## Example
//!
//! ```
//! use agent_framework_core::settings::SecretString;
//!
//! let key = SecretString::new("sk-super-secret");
//! assert_eq!(format!("{key}"), "***");
//! assert_eq!(format!("{key:?}"), "SecretString(\"***\")");
//! assert_eq!(key.expose_secret(), "sk-super-secret");
//! ```

use std::collections::HashMap;
use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// The literal used to mask a [`SecretString`]'s value in [`Debug`]/[`Display`]
/// output.
const MASK: &str = "***";

/// A string wrapper that masks its value when printed via [`Debug`] or
/// [`Display`](std::fmt::Display), to prevent secrets (API keys, tokens,
/// passwords, ...) from
/// accidentally ending up in logs or error messages.
///
/// The real value is still accessible via [`SecretString::expose_secret`],
/// and is preserved (not masked) when (de)serialized with `serde`, since
/// serialization is generally used to persist or transmit the value rather
/// than to display it.
///
/// This is the Rust analogue of upstream's `SecretString(str)`, which masks
/// its `repr()` but not its `str()`/`__format__` behavior. Rust has no
/// implicit string-like coercions, so the equivalent masked-only surface is
/// `Debug` and `Display`; call [`SecretString::expose_secret`] whenever the
/// real value is needed (e.g. to authenticate a request).
#[derive(Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SecretString(String);

impl SecretString {
    /// Wrap `value` as a [`SecretString`].
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Return the real, unmasked value.
    ///
    /// Named to match upstream's `get_secret_value()` / the common Rust
    /// `secrecy`-crate convention, and to make call sites grep-able for
    /// audits of where secrets are actually exposed.
    pub fn expose_secret(&self) -> &str {
        &self.0
    }
}

impl From<String> for SecretString {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for SecretString {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl PartialEq for SecretString {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for SecretString {}

impl fmt::Debug for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SecretString({MASK:?})")
    }
}

impl fmt::Display for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{MASK}")
    }
}

/// Parse `.env`-style file contents into a key/value map.
///
/// Supports simple `KEY=VALUE` lines, blank lines, and `#`-prefixed
/// comments. Surrounding single or double quotes on the value are stripped.
/// This intentionally does not support the fuller dotenv syntax (multiline
/// values, variable expansion, `export` prefixes, ...); it covers the common
/// case without pulling in an extra dependency.
fn parse_dotenv(contents: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        if key.is_empty() {
            continue;
        }
        let mut value = value.trim();
        // Strip a trailing inline comment on unquoted values, e.g. `KEY=value # note`.
        if !(value.starts_with('"') || value.starts_with('\'')) {
            if let Some(idx) = value.find(" #") {
                value = value[..idx].trim();
            }
        }
        let value = if (value.starts_with('"') && value.ends_with('"') && value.len() >= 2)
            || (value.starts_with('\'') && value.ends_with('\'') && value.len() >= 2)
        {
            &value[1..value.len() - 1]
        } else {
            value
        };
        map.insert(key.to_string(), value.to_string());
    }
    map
}

/// Load the key/value pairs from a `.env` file at `path`, if it exists and is
/// readable. Returns an empty map otherwise (missing/unreadable dotenv files
/// are not an error — they simply contribute nothing to resolution).
fn load_dotenv_file(path: &Path) -> HashMap<String, String> {
    std::fs::read_to_string(path)
        .map(|contents| parse_dotenv(&contents))
        .unwrap_or_default()
}

/// Resolve a single setting value using the same precedence as upstream's
/// `load_settings`:
///
/// 1. `override_value` — an explicit value supplied by the caller (e.g. a
///    constructor argument).
/// 2. A `./.env` file in the current working directory, if present, looked
///    up by `key`.
/// 3. The `key` process environment variable.
/// 4. `default`.
///
/// Returns `None` if none of the sources produced a value.
///
/// This is a dependency-free, single-key analogue of upstream's
/// `TypedDict`-driven `load_settings()`; callers needing to resolve several
/// related fields can call this once per field.
pub fn load_setting(
    key: &str,
    override_value: Option<String>,
    default: Option<String>,
) -> Option<String> {
    if let Some(value) = override_value {
        return Some(value);
    }

    let dotenv = load_dotenv_file(Path::new(".env"));
    if let Some(value) = dotenv.get(key) {
        return Some(value.clone());
    }

    if let Ok(value) = std::env::var(key) {
        return Some(value);
    }

    default
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // `std::env::set_var`/`remove_var` mutate global process state, so tests
    // that touch the environment must not run concurrently with each other.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn secret_string_masks_debug_and_display() {
        let secret = SecretString::new("sk-super-secret");
        assert_eq!(format!("{secret:?}"), "SecretString(\"***\")");
        assert_eq!(format!("{secret}"), "***");
        // The mask must never contain the real value as a substring.
        assert!(!format!("{secret:?}").contains("sk-super-secret"));
        assert!(!format!("{secret}").contains("sk-super-secret"));
    }

    #[test]
    fn secret_string_expose_secret_returns_real_value() {
        let secret = SecretString::new("sk-super-secret");
        assert_eq!(secret.expose_secret(), "sk-super-secret");
    }

    #[test]
    fn secret_string_from_conversions() {
        let a: SecretString = "abc".into();
        let b: SecretString = String::from("abc").into();
        assert_eq!(a, b);
        assert_eq!(a.expose_secret(), "abc");
    }

    #[test]
    fn secret_string_equality_compares_real_values() {
        let a = SecretString::new("same");
        let b = SecretString::new("same");
        let c = SecretString::new("different");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn secret_string_clone_preserves_value() {
        let a = SecretString::new("clone-me");
        let b = a.clone();
        assert_eq!(a, b);
        assert_eq!(b.expose_secret(), "clone-me");
    }

    #[test]
    fn secret_string_serde_round_trip_preserves_real_value() {
        let secret = SecretString::new("sk-super-secret");
        let json = serde_json::to_string(&secret).expect("serialize");
        // The serialized form carries the real secret (serialization is not
        // masking) — only Debug/Display mask.
        assert_eq!(json, "\"sk-super-secret\"");
        let round_tripped: SecretString = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(round_tripped, secret);
        assert_eq!(round_tripped.expose_secret(), "sk-super-secret");
    }

    #[test]
    fn parse_dotenv_handles_comments_blanks_and_quotes() {
        let contents = r#"
# a comment
FOO=bar

export BAZ=qux
QUOTED="hello world"
SINGLE='single quoted'
INLINE=value # trailing comment
"#;
        let map = parse_dotenv(contents);
        assert_eq!(map.get("FOO").map(String::as_str), Some("bar"));
        assert_eq!(map.get("BAZ").map(String::as_str), Some("qux"));
        assert_eq!(map.get("QUOTED").map(String::as_str), Some("hello world"));
        assert_eq!(map.get("SINGLE").map(String::as_str), Some("single quoted"));
        assert_eq!(map.get("INLINE").map(String::as_str), Some("value"));
    }

    #[test]
    fn load_setting_override_wins_over_everything() {
        let _guard = ENV_LOCK.lock().unwrap();
        let key = "AF_SETTINGS_TEST_OVERRIDE_WINS";
        std::env::set_var(key, "from-env");

        let result = load_setting(
            key,
            Some("from-override".to_string()),
            Some("from-default".to_string()),
        );

        std::env::remove_var(key);
        assert_eq!(result, Some("from-override".to_string()));
    }

    #[test]
    fn load_setting_env_wins_over_default() {
        let _guard = ENV_LOCK.lock().unwrap();
        let key = "AF_SETTINGS_TEST_ENV_WINS";
        std::env::set_var(key, "from-env");

        let result = load_setting(key, None, Some("from-default".to_string()));

        std::env::remove_var(key);
        assert_eq!(result, Some("from-env".to_string()));
    }

    #[test]
    fn load_setting_falls_back_to_default_when_nothing_else_set() {
        let _guard = ENV_LOCK.lock().unwrap();
        let key = "AF_SETTINGS_TEST_DEFAULT_FALLBACK";
        std::env::remove_var(key); // ensure absent

        let result = load_setting(key, None, Some("from-default".to_string()));
        assert_eq!(result, Some("from-default".to_string()));
    }

    #[test]
    fn load_setting_returns_none_when_nothing_resolves() {
        let _guard = ENV_LOCK.lock().unwrap();
        let key = "AF_SETTINGS_TEST_NOTHING_RESOLVES";
        std::env::remove_var(key); // ensure absent

        let result = load_setting(key, None, None);
        assert_eq!(result, None);
    }

    #[test]
    fn load_setting_dotenv_file_beats_env_but_not_override() {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = std::env::temp_dir().join(format!(
            "af-settings-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let dotenv_path = dir.join(".env");
        std::fs::write(&dotenv_path, "AF_SETTINGS_TEST_DOTENV=from-dotenv\n").unwrap();

        let key = "AF_SETTINGS_TEST_DOTENV";
        std::env::set_var(key, "from-env");

        let original_cwd = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();

        let dotenv_result = load_setting(key, None, None);
        let override_result = load_setting(key, Some("from-override".to_string()), None);

        std::env::set_current_dir(&original_cwd).unwrap();
        std::env::remove_var(key);
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(dotenv_result, Some("from-dotenv".to_string()));
        assert_eq!(override_result, Some("from-override".to_string()));
    }
}
