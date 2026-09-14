//! Turning identifiers into storage keys without letting two of them collide.
//!
//! Ports the derivation upstream added in `_filesystem._storage_key_segment`
//! and applied to its Redis history keys (#8236).
//!
//! # The problem
//!
//! A store that addresses a record by joining identifiers — `{prefix}:{id}` —
//! is only unambiguous while no identifier can contain the separator. Nothing
//! constrains them: a session id is whatever the caller (or the caller's
//! users) put in it. So `prefix="chat"`, `id="a:b"` and `prefix="chat:a"`,
//! `id="b"` address the same key, and two conversations that should be
//! isolated share one list. In a multi-tenant deployment, where the prefix
//! *is* the tenant boundary, that is one tenant reading another's history.
//!
//! [`storage_key_segment`] makes each segment unambiguous before it is
//! joined, so no identifier can impersonate a separator.

/// Whether `value` can be used as a key segment verbatim.
///
/// Deliberately narrow: lowercase ASCII alphanumerics plus `.`, `_` and `-`.
/// Uppercase is excluded so the derivation stays one-to-one even where keys
/// are compared case-insensitively, and `~` is excluded because every encoded
/// segment starts with it — which is what keeps the literal and encoded
/// namespaces from ever overlapping.
fn is_literal_safe(value: &str) -> bool {
    !value.is_empty()
        && value.chars().all(|c| {
            c.is_ascii()
                && ((c.is_ascii_alphanumeric() && !c.is_ascii_uppercase())
                    || matches!(c, '.' | '_' | '-'))
        })
}

/// Render `value` as one unambiguous storage-key segment.
///
/// `encoded_prefix` distinguishes components that might encode the same
/// identifier (a prefix segment and a session segment, say); it must start
/// with `~`, which [`is_literal_safe`] rejects, so an encoded segment can
/// never be mistaken for a literal one.
///
/// A literal-safe value is returned as-is, so the common case — a key prefix
/// like `chat_messages` and a UUID session id — keeps producing exactly the
/// keys it always has, and existing data stays addressable. Everything else
/// is hex-encoded, which is reversible and therefore injective: two different
/// identifiers cannot produce one segment.
///
/// # Divergence: hex, not base32
///
/// Upstream encodes with lowercase base32 and, past a length cap, replaces it
/// with a SHA-256 digest to stay inside filesystem name limits. These are
/// Redis keys rather than path segments — the limit is 512 MB, not 255 bytes
/// — so the cap has nothing to guard, and dropping it keeps the derivation
/// injective all the way up rather than collision-*resistant* past a
/// threshold. Hex over base32 is then just one fewer thing to hand-roll; the
/// two encodings are equally reversible, and the keys are not shared with
/// upstream's implementation in any case (its scoped format differs
/// wholesale).
pub fn storage_key_segment(value: &str, encoded_prefix: &str) -> String {
    debug_assert!(
        encoded_prefix.starts_with('~'),
        "an encoded-segment prefix must start with '~' to stay outside the literal namespace"
    );
    if is_literal_safe(value) {
        return value.to_string();
    }
    let mut out = String::with_capacity(encoded_prefix.len() + value.len() * 2);
    out.push_str(encoded_prefix);
    for byte in value.as_bytes() {
        out.push(char::from_digit((byte >> 4) as u32, 16).unwrap_or('0'));
        out.push(char::from_digit((byte & 0x0f) as u32, 16).unwrap_or('0'));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_safe_values_pass_through_unchanged() {
        // The point of the passthrough: the common case keeps producing the
        // keys it always did, so existing data stays addressable.
        for value in ["chat_messages", "3f2a-9b1c-4d5e", "a.b.c", "x"] {
            assert_eq!(storage_key_segment(value, "~s-"), value);
        }
    }

    #[test]
    fn anything_that_could_be_confused_for_a_separator_is_encoded() {
        for value in ["a:b", "a/b", "has space", "Upper", "~already", ""] {
            let encoded = storage_key_segment(value, "~s-");
            assert!(encoded.starts_with("~s-"), "{value} -> {encoded}");
            assert_ne!(encoded, value);
        }
    }

    #[test]
    fn the_derivation_is_injective_across_the_separator() {
        // The bug this exists for: `{prefix}:{session}` addressed one key for
        // two different (prefix, session) pairs.
        let a = format!(
            "{}:{}",
            storage_key_segment("chat", "~p-"),
            storage_key_segment("a:b", "~s-")
        );
        let b = format!(
            "{}:{}",
            storage_key_segment("chat:a", "~p-"),
            storage_key_segment("b", "~s-")
        );
        assert_ne!(a, b);
        // And without the derivation they are the same string, which is what
        // makes this a real collision rather than a theoretical one.
        assert_eq!(
            format!("{}:{}", "chat", "a:b"),
            format!("{}:{}", "chat:a", "b")
        );
    }

    #[test]
    fn the_component_prefix_keeps_two_components_distinguishable() {
        assert_ne!(
            storage_key_segment("a:b", "~p-"),
            storage_key_segment("a:b", "~s-")
        );
    }

    #[test]
    fn encoding_is_reversible() {
        let encoded = storage_key_segment("a:b/ Ünicode", "~s-");
        let hex = encoded.strip_prefix("~s-").expect("prefixed");
        let bytes: Vec<u8> = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex pair"))
            .collect();
        assert_eq!(String::from_utf8(bytes).unwrap(), "a:b/ Ünicode");
    }

    #[test]
    fn an_encoded_segment_cannot_collide_with_a_literal_one() {
        // A caller whose identifier happens to look like an encoded segment
        // is itself encoded, because `~` is outside the literal alphabet.
        let looks_encoded = storage_key_segment("~s-6162", "~s-");
        let real = storage_key_segment("ab", "~s-");
        assert_ne!(looks_encoded, real);
        assert_eq!(real, "ab");
    }
}
