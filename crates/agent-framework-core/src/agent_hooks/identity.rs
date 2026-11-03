//! Identity providers (§10): binding a record, and an approval, to the
//! exact context it was made over without copying the payload.
//!
//! The default, `jcs-sha256` (§10.2), hashes the RFC 8785 (JSON
//! Canonicalization Scheme) form of the context's *closed* projection: the
//! required core plus the point's conditional fields, each narrowed to the
//! spec's subfields, so optional data (`agent.name`, `usage`, `tools`) never
//! changes the identity. Integers beyond ±(2⁵³−1) are refused rather than
//! rounded (§10.2 input domain): the identity must be injective. This
//! follows the agent-hooks SDK's `canonical` module.

use std::fmt::Write as _;
use std::sync::Arc;

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use super::protocol::{host_error, InterceptionContext};

/// The name of the default identity provider (§10.1).
pub const JCS_SHA256: &str = "jcs-sha256";

/// The largest integer an IEEE-754 double holds exactly (2⁵³−1).
const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

/// A host-supplied identity function.
pub type IdentityFn = Arc<dyn Fn(&InterceptionContext) -> String + Send + Sync>;

/// The host-declared identity provider (§10.1).
#[derive(Clone, Default)]
pub enum IdentityProvider {
    /// RFC 8785 + SHA-256 over the closed projection (`sha256:<hex>`). The
    /// default, as upstream.
    #[default]
    JcsSha256,
    /// Identity-unbound: records carry `null` identities and approvals bind
    /// by correlation only.
    Null,
    /// A host function. Construct with [`IdentityProvider::custom`].
    Custom {
        /// The provider's name, recorded on every record.
        name: String,
        /// The function.
        f: IdentityFn,
    },
}

impl std::fmt::Debug for IdentityProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::JcsSha256 => f.write_str("JcsSha256"),
            Self::Null => f.write_str("Null"),
            Self::Custom { name, .. } => f.debug_struct("Custom").field("name", name).finish(),
        }
    }
}

impl IdentityProvider {
    /// A custom provider. The name must match `^[a-z][a-z0-9_-]*$` and must
    /// not start with `jcs` (reserved for the canonical providers, whose
    /// golden vectors a custom function does not meet).
    pub fn custom(
        name: impl Into<String>,
        f: impl Fn(&InterceptionContext) -> String + Send + Sync + 'static,
    ) -> crate::Result<Self> {
        let name = name.into();
        let mut bytes = name.bytes();
        let valid = bytes.next().is_some_and(|b| b.is_ascii_lowercase())
            && bytes
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-');
        if !valid {
            return Err(crate::Error::Configuration(
                "identity provider names must match ^[a-z][a-z0-9_-]*$".into(),
            ));
        }
        if name.starts_with("jcs") {
            return Err(crate::Error::Configuration(
                "identity provider names beginning with 'jcs' are reserved".into(),
            ));
        }
        Ok(Self::Custom {
            name,
            f: Arc::new(f),
        })
    }

    /// The name recorded as `identity_provider` (`None` when unbound).
    pub fn name(&self) -> Option<String> {
        match self {
            Self::JcsSha256 => Some(JCS_SHA256.into()),
            Self::Null => None,
            Self::Custom { name, .. } => Some(name.clone()),
        }
    }

    /// The identity of `context`: `Ok(None)` when unbound, `Err` (a
    /// payload-free reason and detail) when the provider cannot compute it,
    /// which fails the emission closed.
    pub(crate) fn compute(
        &self,
        context: &InterceptionContext,
    ) -> std::result::Result<Option<String>, (&'static str, String)> {
        match self {
            Self::JcsSha256 => context_identity(context).map(Some),
            Self::Null => Ok(None),
            Self::Custom { f, .. } => {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(context)))
                    .map(Some)
                    .map_err(|_| {
                        (
                            host_error::CONTEXT_INVALID,
                            "identity provider failed".to_string(),
                        )
                    })
            }
        }
    }
}

/// `(field, kept subfields)`; `None` keeps the whole value.
type Keep = (&'static str, Option<&'static [&'static str]>);

const REQUIRED: &[Keep] = &[
    ("spec", None),
    ("interception_point", None),
    ("timestamp", None),
    ("sequence", None),
    ("agent", Some(&["id", "framework"])),
    ("session", Some(&["id"])),
    ("target", None),
];

fn conditional_for(point: &str) -> &'static [Keep] {
    match point {
        "agent_startup" => &[("agent_init", Some(&["tools_registered"]))],
        "input" => &[("input", Some(&["content", "role"]))],
        "pre_model_call" => &[("model", Some(&["id"])), ("messages", None)],
        "post_model_call" => &[
            ("model", Some(&["id"])),
            (
                "response",
                Some(&["content", "tool_calls", "finish_reason"]),
            ),
        ],
        "pre_tool_call" => &[("tool_call", Some(&["id", "name", "args"]))],
        "post_tool_call" => &[
            ("tool_call", Some(&["id", "name", "args"])),
            ("tool_result", Some(&["value", "is_error"])),
        ],
        "output" => &[("output", Some(&["content"]))],
        "agent_shutdown" => &[("summary", Some(&["reason"]))],
        _ => &[],
    }
}

/// The closed required+conditional projection of a context (§10.2).
fn projection(json: &Map<String, Value>) -> Value {
    let point = json
        .get("interception_point")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let mut out = Map::new();
    for (key, keep) in REQUIRED.iter().chain(conditional_for(point)) {
        let Some(value) = json.get(*key) else {
            continue;
        };
        let value = match (keep, value) {
            (Some(keep), Value::Object(inner)) => Value::Object(
                inner
                    .iter()
                    .filter(|(k, _)| keep.contains(&k.as_str()))
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
            ),
            _ => value.clone(),
        };
        out.insert((*key).to_string(), value);
    }
    Value::Object(out)
}

/// Refuse integers outside ±(2⁵³−1): RFC 8785 would round them, making
/// two contexts share an identity. The detail names the path, never the
/// value.
fn check_i_json(value: &Value, path: &str) -> std::result::Result<(), String> {
    match value {
        Value::Number(n) => {
            let out = match (n.as_u64(), n.as_i64()) {
                (Some(u), _) => u > MAX_SAFE_INTEGER,
                (None, Some(i)) => i.unsigned_abs() > MAX_SAFE_INTEGER,
                _ => false,
            };
            if out {
                return Err(format!(
                    "{path}: integer exceeds 2^53; string-encode 64-bit identifiers"
                ));
            }
            Ok(())
        }
        Value::Array(items) => items
            .iter()
            .enumerate()
            .try_for_each(|(i, item)| check_i_json(item, &format!("{path}[{i}]"))),
        Value::Object(map) => map
            .iter()
            .try_for_each(|(k, item)| check_i_json(item, &format!("{path}.{k}"))),
        _ => Ok(()),
    }
}

/// The RFC 8785 canonical form of `value`: object members sorted by their
/// UTF-16 code units, numbers in ECMAScript `Number.prototype.toString`
/// form, and minimal string escapes.
pub fn canonical_json(value: &Value) -> String {
    let mut out = String::new();
    write_canonical(value, &mut out);
    out
}

fn write_canonical(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            // Every number is an IEEE-754 double in JCS.
            let f = n.as_f64().unwrap_or(f64::NAN);
            if f.is_finite() {
                out.push_str(ryu_js::Buffer::new().format_finite(f));
            } else {
                out.push_str("null");
            }
        }
        // serde_json's escaping is RFC 8785's: `"`, `\`, the short escapes
        // and lowercase `\u00xx` for other controls; nothing else.
        Value::String(s) => out.push_str(&serde_json::to_string(s).unwrap_or_default()),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut members: Vec<(&String, &Value)> = map.iter().collect();
            members.sort_by(|(a, _), (b, _)| a.encode_utf16().cmp(b.encode_utf16()));
            out.push('{');
            for (i, (key, item)) in members.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(key).unwrap_or_default());
                out.push(':');
                write_canonical(item, out);
            }
            out.push('}');
        }
    }
}

/// The `jcs-sha256` identity of a context: `sha256:` and the lowercase hex
/// SHA-256 of the canonical closed projection.
pub fn context_identity(
    context: &InterceptionContext,
) -> std::result::Result<String, (&'static str, String)> {
    let preimage = projection(context.as_json());
    check_i_json(&preimage, "$").map_err(|d| (host_error::CONTEXT_INVALID, d))?;
    let digest = Sha256::digest(canonical_json(&preimage).as_bytes());
    let mut out = String::with_capacity(71);
    out.push_str("sha256:");
    for b in digest {
        let _ = write!(out, "{b:02x}");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    #[allow(clippy::excessive_precision)]
    fn canonical_form_follows_rfc_8785() {
        // Members sort by UTF-16 code units: U+10000 (a surrogate pair)
        // precedes U+E000.
        let v = json!({"\u{e000}": 1, "\u{10000}": 2, "a": [1.0, 0.5, 1e21, -0.0], "b": "\u{1}\""});
        assert_eq!(
            canonical_json(&v),
            "{\"a\":[1,0.5,1e+21,0],\"b\":\"\\u0001\\\"\",\"\u{10000}\":2,\"\u{e000}\":1}"
        );
        // RFC 8785 appendix B number samples.
        for (n, s) in [
            (333333333.33333329, "333333333.3333333"),
            (1e30, "1e+30"),
            (4.5, "4.5"),
            (2e-3, "0.002"),
            (1e-7, "1e-7"),
        ] {
            assert_eq!(canonical_json(&json!(n)), s);
        }
    }

    #[test]
    fn custom_names_are_checked() {
        assert!(IdentityProvider::custom("mine", |_| String::new()).is_ok());
        assert!(IdentityProvider::custom("jcs-x", |_| String::new()).is_err());
        assert!(IdentityProvider::custom("Mine", |_| String::new()).is_err());
        assert!(IdentityProvider::custom("", |_| String::new()).is_err());
    }
}
