//! Per-request platform identity: the request context the platform sends as
//! headers, and the trusted scope hosted state is keyed by.

use agent_framework_core::error::{Error, Result};
use axum::http::{HeaderMap, HeaderName, HeaderValue};
use sha2::{Digest, Sha256};

use crate::config::FoundryHostConfig;

/// Platform header names.
pub mod headers {
    /// Request correlation id (echoed on every response).
    pub const REQUEST_ID: &str = "x-request-id";
    /// The server SDK stack, set on every response.
    pub const SERVER_VERSION: &str = "x-platform-server";
    /// The resolved agent session id.
    pub const SESSION_ID: &str = "x-agent-session-id";
    /// The platform-injected user id (trusted only when hosted).
    pub const USER_ID: &str = "x-agent-user-id";
    /// The platform-minted opaque call id, forwarded on outbound 1P calls.
    pub const FOUNDRY_CALL_ID: &str = "x-agent-foundry-call-id";
    /// The invocation id (Invocations protocol).
    pub const INVOCATION_ID: &str = "x-agent-invocation-id";
    /// Classifies an error response (`user`, `platform`, `upstream`).
    pub const ERROR_SOURCE: &str = "x-platform-error-source";
    /// Diagnostic detail on a `platform` error response.
    pub const ERROR_DETAIL: &str = "x-platform-error-detail";
    /// The platform's original-subject assertions (upstream's draft
    /// header); see [`FoundryOriginalSubject`](super::FoundryOriginalSubject).
    pub const ORIGINAL_SUBJECT: &str = "x-ms-original-subject";
    /// Foundry baggage for a model endpoint.
    pub const FOUNDRY_BAGGAGE: &str = "x-ms-foundry-baggage";
}

/// What the platform said about one request (upstream
/// `FoundryAgentRequestContext`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FoundryRequestContext {
    /// The request id (`x-request-id`), generated when absent.
    pub request_id: String,
    /// The caller-visible session id, if any.
    pub session_id: Option<String>,
    /// `x-agent-user-id`.
    pub user_id: Option<String>,
    /// `x-agent-foundry-call-id`.
    pub call_id: Option<String>,
    /// `x-ms-original-subject`: invocation-local attribution assertions
    /// (experimental upstream); `None` when the header is absent.
    pub original_subject: Option<FoundryOriginalSubject>,
}

fn header(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_string)
}

impl FoundryRequestContext {
    /// Read the context from request headers; `session_id` is the caller's
    /// (a query parameter or body field) when the protocol has one.
    ///
    /// Fails when `x-ms-original-subject` is present but malformed (see
    /// [`FoundryOriginalSubject::from_headers`]); the hosts then reject the
    /// request before any handler runs.
    pub fn from_headers(
        headers: &HeaderMap,
        session_id: Option<String>,
    ) -> std::result::Result<Self, InvalidOriginalSubject> {
        let original_subject = FoundryOriginalSubject::from_headers(headers)?;
        Ok(Self {
            original_subject,
            ..Self::identity_from_headers(headers, session_id)
        })
    }

    /// The context without the original subject, for routes where upstream
    /// does not read the header.
    pub(crate) fn identity_from_headers(headers: &HeaderMap, session_id: Option<String>) -> Self {
        Self {
            request_id: header(headers, headers::REQUEST_ID)
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
            session_id: session_id
                .filter(|s| !s.is_empty())
                .or_else(|| header(headers, headers::SESSION_ID)),
            user_id: header(headers, headers::USER_ID),
            call_id: header(headers, headers::FOUNDRY_CALL_ID),
            original_subject: None,
        }
    }

    /// Per-call headers for a model endpoint whose owner has enabled the
    /// draft original-subject contract (experimental upstream:
    /// `FoundryAgentRequestContext.model_attribution_headers`).
    ///
    /// `headers` (existing per-call headers, including any Foundry baggage)
    /// are copied, not mutated. Inbound `x-ms-original-subject*` headers and
    /// `x-agent-foundry-call-id` are dropped, as is any reserved
    /// `foundry.original_subject` baggage member. With a subject, its whole
    /// quoted dictionary is emitted once, percent-encoded, as the first
    /// `x-ms-foundry-baggage` member, and the call id is added when known;
    /// without one, only the unrelated baggage is kept.
    ///
    /// Use the result per call, never as a client default, and never send
    /// it to external endpoints, storage or tools. Fails on a partial, empty
    /// or non-visible-ASCII tuple, a quoted tuple over 256 characters, or
    /// merged baggage over 1024 characters.
    pub fn model_attribution_headers(
        &self,
        headers: &HeaderMap,
    ) -> std::result::Result<HeaderMap, InvalidOriginalSubject> {
        let mut result = HeaderMap::new();
        let mut baggage: Vec<String> = Vec::new();
        let subject_prefix = format!("{}-", headers::ORIGINAL_SUBJECT);
        let reserved_prefix = format!("{ORIGINAL_SUBJECT_BAGGAGE}.");
        for (name, value) in headers {
            let lowered = name.as_str();
            if lowered == headers::FOUNDRY_BAGGAGE {
                let Ok(value) = value.to_str() else {
                    continue;
                };
                for member in value.split(',') {
                    let member = member.trim();
                    let raw_key = member
                        .split(';')
                        .next()
                        .unwrap_or_default()
                        .split('=')
                        .next()
                        .unwrap_or_default()
                        .trim();
                    let key = percent_decode(raw_key).to_lowercase();
                    if member.is_empty()
                        || key == ORIGINAL_SUBJECT_BAGGAGE
                        || key.starts_with(&reserved_prefix)
                    {
                        continue;
                    }
                    baggage.push(member.to_string());
                }
            } else if lowered != headers::ORIGINAL_SUBJECT
                && !lowered.starts_with(&subject_prefix)
                && lowered != headers::FOUNDRY_CALL_ID
            {
                result.append(name.clone(), value.clone());
            }
        }
        let baggage_name = HeaderName::from_static(headers::FOUNDRY_BAGGAGE);
        let Some(subject) = &self.original_subject else {
            if let Some(value) = (!baggage.is_empty())
                .then(|| HeaderValue::from_str(&baggage.join(",")).ok())
                .flatten()
            {
                result.insert(baggage_name, value);
            }
            return Ok(result);
        };
        let mut members = Vec::with_capacity(3);
        for (name, value) in [
            ("oid", &subject.oid),
            ("tid", &subject.tid),
            ("idtype", &subject.idtype),
        ] {
            let value = value
                .as_deref()
                .filter(|v| !v.is_empty() && v.chars().all(|c| ('!'..='~').contains(&c)))
                .ok_or(InvalidOriginalSubject(
                    "Original-subject attribution requires a complete, HTTP-safe tuple.",
                ))?;
            let escaped = value.replace('\\', r"\\").replace('"', r#"\""#);
            members.push(format!(r#"{name}="{escaped}""#));
        }
        let value = members.join(", ");
        if value.len() > MAX_ORIGINAL_SUBJECT_LENGTH {
            return Err(InvalidOriginalSubject(
                "Original-subject attribution requires a quoted tuple of at most 256 characters.",
            ));
        }
        // Reserve attribution space before unrelated members can expand.
        baggage.insert(
            0,
            format!("{ORIGINAL_SUBJECT_BAGGAGE}={}", percent_encode(&value)),
        );
        let merged = baggage.join(",");
        let too_long = InvalidOriginalSubject(
            "Original-subject attribution requires merged baggage of at most 1024 characters.",
        );
        if merged.chars().count() > MAX_FOUNDRY_BAGGAGE_LENGTH {
            return Err(too_long);
        }
        result.insert(
            baggage_name,
            HeaderValue::from_str(&merged).map_err(|_| too_long)?,
        );
        if let Some(call_id) = self
            .call_id
            .as_deref()
            .and_then(|c| HeaderValue::from_str(c).ok())
        {
            result.insert(HeaderName::from_static(headers::FOUNDRY_CALL_ID), call_id);
        }
        Ok(result)
    }
}

/// The baggage member that carries the original subject.
const ORIGINAL_SUBJECT_BAGGAGE: &str = "foundry.original_subject";
const MAX_ORIGINAL_SUBJECT_LENGTH: usize = 256;
const MAX_FOUNDRY_BAGGAGE_LENGTH: usize = 1024;

/// A malformed `x-ms-original-subject` header, or a subject that cannot be
/// sent as model attribution (upstream raises `ValueError` with the same
/// messages).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidOriginalSubject(&'static str);

impl InvalidOriginalSubject {
    /// The reason, worded as upstream.
    pub fn message(&self) -> &'static str {
        self.0
    }
}

impl std::fmt::Display for InvalidOriginalSubject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for InvalidOriginalSubject {}

const NOT_UNIQUE_MEMBERS: InvalidOriginalSubject = InvalidOriginalSubject(
    "Original-subject attribution requires unique oid/tid/idtype string members.",
);

/// Invocation-local original-subject assertions from the platform's
/// `x-ms-original-subject` header, separate from model authentication
/// (experimental upstream: `FoundryOriginalSubject`).
///
/// These values are not proof of identity: the receiving model service
/// must establish provenance and invocation binding before using them.
/// Omitted keys stay `None` and empty strings stay empty; nothing is
/// normalized or inferred. `Debug` does not print the values.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct FoundryOriginalSubject {
    /// The original subject's object id (not the agent's or the session
    /// owner's).
    pub oid: Option<String>,
    /// The original subject's tenant id.
    pub tid: Option<String>,
    /// The original subject's classification, preserved as sent.
    pub idtype: Option<String>,
}

impl std::fmt::Debug for FoundryOriginalSubject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FoundryOriginalSubject")
            .finish_non_exhaustive()
    }
}

impl FoundryOriginalSubject {
    /// Parse the header's draft RFC 9651 dictionary profile: only `oid`,
    /// `tid` and `idtype` string members, each at most once, in any order,
    /// without parameters. `Ok(None)` when the header is absent; an error
    /// for a repeated header field, an empty dictionary, an unknown or
    /// duplicate member, a parameter, or any other malformed syntax.
    pub fn from_headers(
        headers: &HeaderMap,
    ) -> std::result::Result<Option<Self>, InvalidOriginalSubject> {
        let mut fields = headers.get_all(headers::ORIGINAL_SUBJECT).iter();
        let Some(field) = fields.next() else {
            return Ok(None);
        };
        if fields.next().is_some() {
            return Err(InvalidOriginalSubject(
                "Original-subject attribution requires a single header field.",
            ));
        }
        let mut remaining = trim_ows(field.as_bytes());
        let mut subject = Self::default();
        let mut any = false;
        while !remaining.is_empty() {
            let (key, value, rest) = parse_member(remaining).ok_or(NOT_UNIQUE_MEMBERS)?;
            let slot = match key {
                "oid" => &mut subject.oid,
                "tid" => &mut subject.tid,
                _ => &mut subject.idtype,
            };
            if slot.is_some() {
                return Err(NOT_UNIQUE_MEMBERS);
            }
            *slot = Some(value);
            any = true;
            remaining = trim_ows_start(rest);
            if remaining.is_empty() {
                break;
            }
            match remaining.split_first() {
                Some((b',', after)) if !trim_ows(after).is_empty() => {
                    remaining = trim_ows_start(after);
                }
                _ => {
                    return Err(InvalidOriginalSubject(
                        "Original-subject attribution requires comma-separated dictionary members.",
                    ))
                }
            }
        }
        if !any {
            return Err(InvalidOriginalSubject(
                "Original-subject attribution requires a nonempty dictionary.",
            ));
        }
        Ok(Some(subject))
    }
}

fn trim_ows_start(bytes: &[u8]) -> &[u8] {
    let start = bytes
        .iter()
        .position(|b| !matches!(b, b' ' | b'\t'))
        .unwrap_or(bytes.len());
    &bytes[start..]
}

fn trim_ows(bytes: &[u8]) -> &[u8] {
    let bytes = trim_ows_start(bytes);
    let end = bytes
        .iter()
        .rposition(|b| !matches!(b, b' ' | b'\t'))
        .map_or(0, |i| i + 1);
    &bytes[..end]
}

/// One `key="value"` member at the start of `input`, as upstream's
/// `(oid|tid|idtype)="((?:[\x20-\x21\x23-\x5b\x5d-\x7e]|\\["\\])*)"`
/// matches it: the key, the unescaped value, and what follows.
fn parse_member(input: &[u8]) -> Option<(&'static str, String, &[u8])> {
    let (key, rest) = ["oid", "tid", "idtype"]
        .into_iter()
        .find_map(|k| Some((k, input.strip_prefix(k.as_bytes())?)))?;
    let mut rest = rest.strip_prefix(b"=\"")?;
    let mut value = String::new();
    loop {
        match rest.split_first()? {
            (b'"', after) => return Some((key, value, after)),
            (b'\\', after) => {
                let (&escaped, after) = after.split_first()?;
                if !matches!(escaped, b'"' | b'\\') {
                    return None;
                }
                value.push(escaped as char);
                rest = after;
            }
            (&c, after) if (0x20..=0x7e).contains(&c) => {
                value.push(c as char);
                rest = after;
            }
            _ => return None,
        }
    }
}

/// Percent-encode all but RFC 3986 unreserved characters (Python's
/// `quote(value, safe='')`).
fn percent_encode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for b in value.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Decode `%XX` escapes, leaving malformed ones as they are (Python's
/// `unquote`).
fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            if let Some(byte) = value
                .get(i + 1..i + 3)
                .filter(|h| h.bytes().all(|b| b.is_ascii_hexdigit()))
                .and_then(|h| u8::from_str_radix(h, 16).ok())
            {
                out.push(byte);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The trusted identity hosted state is partitioned by: the platform
/// sandbox session and user, separate from caller-facing continuation ids
/// (upstream `FoundryRequestScope`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FoundryRequestScope {
    /// The sandbox session id.
    pub session_id: String,
    /// The platform user id.
    pub user_id: Option<String>,
    /// The platform call id.
    pub call_id: Option<String>,
    /// Whether the host runs in Foundry.
    pub is_hosted: bool,
}

impl FoundryRequestScope {
    /// Resolve the scope. Hosted, the session comes from the platform's
    /// `FOUNDRY_AGENT_SESSION_ID` (a caller-supplied one must match it) and
    /// a user id and call id are required; locally, the request's session id
    /// or `local_session_id` is used.
    pub fn from_context(
        config: &FoundryHostConfig,
        context: &FoundryRequestContext,
        local_session_id: Option<&str>,
    ) -> Result<Self> {
        let session_id = if config.is_hosted {
            if config.session_id.trim().is_empty() {
                return Err(Error::Configuration(
                    "Foundry hosted requests require a platform FOUNDRY_AGENT_SESSION_ID.".into(),
                ));
            }
            if context
                .session_id
                .as_deref()
                .is_some_and(|s| s != config.session_id)
            {
                return Err(Error::Configuration(
                    "The request agent_session_id does not match the platform session ID.".into(),
                ));
            }
            if context.user_id.is_none() || context.call_id.is_none() {
                return Err(Error::Configuration(
                    "Foundry hosted requests require a trusted user ID and call ID.".into(),
                ));
            }
            config.session_id.clone()
        } else {
            context
                .session_id
                .clone()
                .or_else(|| local_session_id.map(str::to_string))
                .ok_or_else(|| {
                    Error::Configuration(
                        "A Foundry agent session ID is required to handle the request.".into(),
                    )
                })?
        };
        Ok(Self {
            session_id,
            user_id: context.user_id.clone(),
            call_id: context.call_id.clone(),
            is_hosted: config.is_hosted,
        })
    }

    /// The SHA-256 (hex) of the framed `[user_id, session_id]` identity, so
    /// store names reveal neither id.
    pub fn storage_key(&self) -> String {
        let framed = serde_json::to_string(&serde_json::json!([self.user_id, self.session_id]))
            .unwrap_or_default();
        hex(&Sha256::digest(framed.as_bytes()))
    }
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The state-store name for `root` under `scope` (upstream `_store_name`):
/// `root[/context]` locally, `root/v2/<storage key>[/<hashed context>]`
/// hosted.
pub fn store_name(
    root: &str,
    scope: Option<&FoundryRequestScope>,
    context_id: Option<&str>,
) -> String {
    match scope {
        None => match context_id {
            Some(id) => format!("{root}/{id}"),
            None => root.to_string(),
        },
        Some(scope) => {
            let mut name = format!("{root}/v2/{}", scope.storage_key());
            if let Some(id) = context_id {
                name.push('/');
                name.push_str(&hex(&Sha256::digest(id.as_bytes()))[..24]);
            }
            name
        }
    }
}

/// Ids in the agentserver's `{prefix}_{partition key}{entropy}` format: an
/// 18-character partition key (16 hex digits and `00`) followed by 32
/// alphanumeric characters (upstream `IdGenerator`).
pub mod ids {
    fn entropy() -> String {
        format!(
            "{}{}",
            uuid::Uuid::new_v4().simple(),
            uuid::Uuid::new_v4().simple()
        )[..32]
            .to_string()
    }

    /// The partition key embedded in `id`, if it has one (new 50-character
    /// bodies carry it first; legacy 48-character bodies carry 16 characters
    /// last, which are suffixed with `00`).
    pub fn partition_key(id: &str) -> Option<String> {
        let (_, body) = id.split_once('_')?;
        match body.len() {
            50 => Some(body[..18].to_string()),
            48 => Some(format!("{}00", &body[32..])),
            _ => None,
        }
    }

    /// A new id with `prefix`, co-located with `hint`'s partition when it
    /// has one.
    pub fn new_id(prefix: &str, hint: Option<&str>) -> String {
        let key = hint
            .and_then(partition_key)
            .unwrap_or_else(|| format!("{}00", &uuid::Uuid::new_v4().simple().to_string()[..16]));
        format!("{prefix}_{key}{}", entropy())
    }

    /// A response id (`caresp_…`).
    pub fn response_id(hint: Option<&str>) -> String {
        new_id("caresp", hint)
    }

    /// A message item id (`msg_…`).
    pub fn message_id(hint: Option<&str>) -> String {
        new_id("msg", hint)
    }

    /// A function-call item id (`fc_…`).
    pub fn function_call_id(hint: Option<&str>) -> String {
        new_id("fc", hint)
    }
}
