//! Per-call `x-client-*` request headers for a Foundry run.
//!
//! The Foundry platform forwards headers prefixed `x-client-` transparently
//! from the Agent Endpoint into the agent container, which makes them the
//! channel for something the *platform in front of the model* must read —
//! most usefully the identity of the end user a run is being made on behalf
//! of (`x-client-end-user-id`), attested per run rather than per client.
//!
//! That "per run" is the whole point, and it is why the headers travel on
//! [`ChatOptions`] rather than on the client: one `FoundryChatClient` serves
//! every user of a multi-tenant SaaS, so an identity pinned at construction
//! would be the wrong one for all but the first caller.
//!
//! ```no_run
//! use agent_framework_core::types::ChatOptions;
//! use agent_framework_foundry::FoundryClientHeaders;
//!
//! # fn demo() -> agent_framework_core::error::Result<()> {
//! let options = ChatOptions::new()
//!     .with_client_header("x-client-end-user-id", "user-42")?
//!     .with_client_header("x-client-request-id", "7f3c…")?;
//! # let _ = options;
//! # Ok(())
//! # }
//! ```
//!
//! # Per-run options replace, rather than merge with, a client-level set
//!
//! `ChatOptions::merge` combines `additional_properties` with a map
//! `extend`, so for any one key the per-run value wins whole. The carrier is
//! one such key: an agent built with `x-client-app-version` in its default
//! options, run with `AgentRunOptions` carrying `x-client-end-user-id`, sends
//! only the second. This is upstream's behaviour too — its dictionary lives
//! on one `ChatOptions` instance — but it is worth stating, because the
//! headers *look* additive and the loss is silent.
//!
//! Stamp the whole set the run should carry onto the run's own options, or
//! read the client-level ones back with
//! [`client_headers`](FoundryClientHeaders::client_headers) and re-stamp them
//! alongside.
//!
//! # Divergence from upstream
//!
//! Upstream .NET needs an agent decorator and a transport policy for this
//! (`AIAgentBuilder.UseClientHeaders()` plus an `OpenAIRequestPolicies`
//! registration), and documents the call as a **silent no-op** when either is
//! missing or when the underlying client is not OpenAI-backed. Neither piece
//! is needed here: `ChatOptions` already reaches
//! [`FoundryChatClient`](crate::FoundryChatClient)'s transport, so a stamped
//! header is delivered by construction, and a malformed one is an error
//! rather than silence. A port of the no-op would reproduce the one property
//! of upstream's design its own documentation warns about.
//!
//! Upstream also validates these headers in its stamping API only (its
//! carrier key is `internal`, so nothing else can reach the dictionary). The
//! carrier here is a public map on a public struct, so the transport
//! validates too — see [`CLIENT_HEADERS_PROPERTY`]. The two checks are not
//! redundant: this one is the early, specific error that names the Foundry
//! prefix rule; that one is what the client can actually rely on.

use agent_framework_azure::responses::CLIENT_HEADERS_PROPERTY;
use agent_framework_core::error::{Error, Result};
use agent_framework_core::types::ChatOptions;
use serde_json::{Map, Value};

/// The prefix every forwarded client header must carry, matched
/// case-insensitively. Anything else is dropped by the platform rather than
/// forwarded, so sending it would be a silent no-op at the far end.
pub const CLIENT_HEADER_PREFIX: &str = "x-client-";

/// Stamp per-call `x-client-*` headers onto a [`ChatOptions`].
///
/// Implemented for `ChatOptions`; see the [module docs](self).
pub trait FoundryClientHeaders: Sized {
    /// Add one `x-client-*` header to this call.
    ///
    /// Replaces any header already stamped under the same name, compared
    /// case-insensitively — HTTP header names are case-insensitive, so
    /// keeping both would leave which one wins to the transport.
    ///
    /// # Errors
    /// [`Error::Configuration`] when `name` does not start with
    /// [`CLIENT_HEADER_PREFIX`], when either `name` or `value` is empty, or
    /// when either contains a NUL, carriage return or line feed.
    fn with_client_header(self, name: impl AsRef<str>, value: impl Into<String>) -> Result<Self>;

    /// Add several `x-client-*` headers to this call.
    ///
    /// All-or-nothing: if any pair is invalid the options are returned
    /// unchanged by the error, so a partially applied set cannot reach the
    /// wire. Upstream stages its pairs for the same reason.
    ///
    /// # Errors
    /// As [`with_client_header`](Self::with_client_header), for the first
    /// invalid pair.
    fn with_client_headers<I, N, V>(self, headers: I) -> Result<Self>
    where
        I: IntoIterator<Item = (N, V)>,
        N: AsRef<str>,
        V: Into<String>;

    /// The `x-client-*` headers stamped on this call, in no particular order.
    /// Empty when none are.
    fn client_headers(&self) -> Vec<(&str, &str)>;
}

impl FoundryClientHeaders for ChatOptions {
    fn with_client_header(self, name: impl AsRef<str>, value: impl Into<String>) -> Result<Self> {
        self.with_client_headers([(name.as_ref().to_string(), value.into())])
    }

    fn with_client_headers<I, N, V>(mut self, headers: I) -> Result<Self>
    where
        I: IntoIterator<Item = (N, V)>,
        N: AsRef<str>,
        V: Into<String>,
    {
        // Staged, so a later invalid pair cannot leave the earlier ones
        // stamped on options the caller is about to run with.
        let mut staged = Vec::new();
        for (name, value) in headers {
            let name = name.as_ref();
            let value = value.into();
            validate(name, &value)?;
            staged.push((name.to_string(), value));
        }
        if staged.is_empty() {
            return Ok(self);
        }
        let carrier = self
            .additional_properties
            .entry(CLIENT_HEADERS_PROPERTY.to_string())
            .or_insert_with(|| Value::Object(Map::new()));
        // A foreign value in the slot is the caller's own doing, and silently
        // discarding whatever they put there would be worse than refusing.
        let carrier = carrier.as_object_mut().ok_or_else(|| {
            Error::Configuration(format!(
                "additional_properties[\"{CLIENT_HEADERS_PROPERTY}\"] is occupied by a value \
                 that is not an object of header name/value strings"
            ))
        })?;
        for (name, value) in staged {
            // Case-insensitive replace: `X-Client-Id` and `x-client-id` are
            // one header, so a second stamping must not add a duplicate the
            // transport then has to choose between.
            if let Some(existing) = carrier
                .keys()
                .find(|k| k.eq_ignore_ascii_case(&name))
                .cloned()
            {
                carrier.remove(&existing);
            }
            carrier.insert(name, Value::String(value));
        }
        Ok(self)
    }

    fn client_headers(&self) -> Vec<(&str, &str)> {
        self.additional_properties
            .get(CLIENT_HEADERS_PROPERTY)
            .and_then(Value::as_object)
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| v.as_str().map(|v| (k.as_str(), v)))
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// The Foundry platform's rules for a forwarded header, checked before the
/// pair is stamped.
///
/// The NUL/CR/LF refusals are the ones upstream added in #8847. They are not
/// load-bearing against header injection here — `HeaderName`/`HeaderValue`
/// parsing in the transport cannot be talked into accepting any of the three,
/// so the request would fail rather than split — but they turn a failure
/// discovered at `send()` into an error that names the offending header at
/// the point the caller wrote it.
fn validate(name: &str, value: &str) -> Result<()> {
    if name.trim().is_empty() {
        return Err(Error::Configuration(
            "a client header name must not be empty or whitespace".into(),
        ));
    }
    if contains_transport_delimiter(name) {
        return Err(Error::Configuration(format!(
            "client header name '{}' must not contain a NUL, carriage return or line feed",
            name.escape_debug()
        )));
    }
    if value.is_empty() {
        return Err(Error::Configuration(format!(
            "client header '{name}' must have a non-empty value"
        )));
    }
    if contains_transport_delimiter(value) {
        // The value is not echoed: it is the field most likely to hold an
        // end-user identifier, and an error message is the wrong place for
        // one.
        return Err(Error::Configuration(format!(
            "the value of client header '{name}' must not contain a NUL, carriage return or \
             line feed"
        )));
    }
    if !name
        .get(..CLIENT_HEADER_PREFIX.len())
        .is_some_and(|p| p.eq_ignore_ascii_case(CLIENT_HEADER_PREFIX))
    {
        return Err(Error::Configuration(format!(
            "client header name '{name}' must start with '{CLIENT_HEADER_PREFIX}' \
             (case-insensitive): the Foundry platform forwards no other header into the agent \
             container, so one named otherwise would be dropped rather than delivered"
        )));
    }
    Ok(())
}

/// Whether `s` holds a character that terminates a field on the wire. Checked
/// on the name before it is used in an error message, as upstream does.
fn contains_transport_delimiter(s: &str) -> bool {
    s.contains(['\0', '\r', '\n'])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers_of(options: &ChatOptions) -> Vec<(String, String)> {
        let mut v: Vec<(String, String)> = options
            .client_headers()
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        v.sort();
        v
    }

    #[test]
    fn a_valid_header_is_stamped_and_read_back() {
        let options = ChatOptions::new()
            .with_client_header("x-client-end-user-id", "user-42")
            .unwrap();
        assert_eq!(
            headers_of(&options),
            vec![("x-client-end-user-id".to_string(), "user-42".to_string())]
        );
    }

    #[test]
    fn the_prefix_is_required_but_matched_case_insensitively() {
        // The platform forwards nothing else, so a header named otherwise
        // would be dropped at the far end rather than delivered.
        let err = ChatOptions::new()
            .with_client_header("x-tenant-id", "acme")
            .expect_err("a header outside the forwarded prefix must be refused");
        assert!(err.to_string().contains("must start with 'x-client-'"));
        // Upstream matches the prefix case-insensitively; so does this.
        for name in ["X-Client-Id", "X-CLIENT-ID", "x-client-id"] {
            ChatOptions::new()
                .with_client_header(name, "v")
                .unwrap_or_else(|e| panic!("{name} should be accepted: {e}"));
        }
    }

    #[test]
    fn a_name_shorter_than_the_prefix_is_refused_rather_than_panicking() {
        // `&name[..prefix.len()]` would panic here; the slice is fallible for
        // this reason, and a short name is simply not prefixed.
        for name in ["x", "x-client", "x-clien"] {
            let err = ChatOptions::new()
                .with_client_header(name, "v")
                .expect_err("not prefixed");
            assert!(err.to_string().contains("must start with"), "{name}");
        }
    }

    #[test]
    fn an_empty_name_or_value_is_refused() {
        assert!(ChatOptions::new()
            .with_client_header("", "v")
            .expect_err("empty name")
            .to_string()
            .contains("must not be empty"));
        assert!(ChatOptions::new()
            .with_client_header("   ", "v")
            .expect_err("whitespace name")
            .to_string()
            .contains("must not be empty"));
        // Upstream refuses an empty value too: a header present with no value
        // says something different from one not sent, and never the thing the
        // caller meant.
        assert!(ChatOptions::new()
            .with_client_header("x-client-id", "")
            .expect_err("empty value")
            .to_string()
            .contains("non-empty value"));
    }

    #[test]
    fn a_transport_delimiter_is_refused_in_either_half() {
        for bad in ["x-client-a\rb", "x-client-a\nb", "x-client-a\0b"] {
            assert!(
                ChatOptions::new()
                    .with_client_header(bad, "v")
                    .expect_err("delimiter in name")
                    .to_string()
                    .contains("carriage return or line feed"),
                "{bad:?}"
            );
        }
        for bad in ["a\rb", "a\nb", "a\0b", "v\r\nx-injected: 1"] {
            let err = ChatOptions::new()
                .with_client_header("x-client-id", bad)
                .expect_err("delimiter in value");
            let msg = err.to_string();
            assert!(msg.contains("carriage return or line feed"), "{bad:?}");
            // The value is not echoed back: it is the field most likely to
            // carry an end-user identifier.
            assert!(!msg.contains("x-injected"), "the value leaked: {msg}");
        }
    }

    #[test]
    fn re_stamping_a_name_replaces_it_regardless_of_case() {
        let options = ChatOptions::new()
            .with_client_header("X-Client-Id", "first")
            .unwrap()
            .with_client_header("x-client-id", "second")
            .unwrap();
        // One header, not two: HTTP names compare case-insensitively, so
        // keeping both would leave the transport to pick a winner.
        assert_eq!(
            headers_of(&options),
            vec![("x-client-id".to_string(), "second".to_string())]
        );
    }

    #[test]
    fn a_batch_is_all_or_nothing() {
        let err = ChatOptions::new()
            .with_client_headers([("x-client-a", "1"), ("nope", "2")])
            .expect_err("the second pair is invalid");
        assert!(err.to_string().contains("must start with"));
        // The error consumed the options, so there is nothing half-stamped to
        // run with — which is the property, not an accident of the signature.
        let options = ChatOptions::new()
            .with_client_headers([("x-client-a", "1"), ("x-client-b", "2")])
            .unwrap();
        assert_eq!(
            headers_of(&options),
            vec![
                ("x-client-a".to_string(), "1".to_string()),
                ("x-client-b".to_string(), "2".to_string())
            ]
        );
    }

    #[test]
    fn an_empty_batch_adds_no_carrier_at_all() {
        // Not merely "no headers": an empty carrier object would be a body
        // field the transport has to know to strip, for no gain.
        let options = ChatOptions::new()
            .with_client_headers(Vec::<(String, String)>::new())
            .unwrap();
        assert!(!options
            .additional_properties
            .contains_key(CLIENT_HEADERS_PROPERTY));
        assert!(options.client_headers().is_empty());
    }

    #[test]
    fn a_foreign_value_in_the_carrier_slot_is_refused_not_discarded() {
        let mut options = ChatOptions::new();
        options.additional_properties.insert(
            CLIENT_HEADERS_PROPERTY.to_string(),
            Value::String("x".into()),
        );
        let err = options
            .with_client_header("x-client-id", "v")
            .expect_err("a foreign carrier value must be refused");
        assert!(err.to_string().contains("occupied by a value"));
    }

    #[test]
    fn no_headers_reads_back_empty() {
        assert!(ChatOptions::new().client_headers().is_empty());
    }
}
