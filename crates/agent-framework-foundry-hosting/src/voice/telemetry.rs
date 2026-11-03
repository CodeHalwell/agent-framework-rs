//! Voice trace-context propagation, spans and metrics (upstream
//! `voice/_tracing.py` and the propagation helpers of `_voice_host.py`).
//!
//! Spans are `tracing` spans; with the `otel` feature the connection span
//! joins the caller's W3C trace, carries the allow-listed baggage, and the
//! connection and target durations and propagation failures are recorded
//! as OpenTelemetry metrics on the `azure.ai.agentserver.invocations.voice`
//! meter.

use std::sync::LazyLock;
use std::time::Duration;

use axum::http::HeaderMap;
use regex::Regex;

/// The instrumentation scope.
#[cfg_attr(not(feature = "otel"), allow(dead_code))]
pub(crate) const SCOPE_NAME: &str = "azure.ai.agentserver.invocations.voice";
#[cfg_attr(not(feature = "otel"), allow(dead_code))]
const SCHEMA_URL: &str = "https://opentelemetry.io/schemas/gen-ai-dev/1.42.0-dev";

/// The baggage members a Voice connection keeps.
pub(crate) const BAGGAGE_KEYS: &[&str] = &[
    "azure.ai.agentserver.session_id",
    "microsoft.a365.agent.blueprint.id",
    "user.id",
    "gen_ai.agent.id",
    "microsoft.tenant.id",
];
pub(crate) const SESSION_ID_BAGGAGE_KEY: &str = "azure.ai.agentserver.session_id";

static SESSION_ID: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[A-Za-z0-9_-]{8,128}$").expect("valid regex"));
static OPAQUE_ID: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[A-Za-z0-9][A-Za-z0-9._:@/-]*$").expect("valid regex"));
static PROTOCOL_VERSION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[0-9]{1,3}(?:\.[0-9]{1,3}){1,2}$").expect("valid regex"));
static TRACESTATE_MEMBER: LazyLock<Regex> = LazyLock::new(|| {
    let key = r"[a-z][_0-9a-z\-*/]{0,255}|[a-z0-9][_0-9a-z\-*/]{0,240}@[a-z][_0-9a-z\-*/]{0,13}";
    let value = r"[\x20-\x2b\x2d-\x3c\x3e-\x7e]{0,255}[\x21-\x2b\x2d-\x3c\x3e-\x7e]";
    Regex::new(&format!(r"^({key})=({value})[ \t]*$")).expect("valid regex")
});
static TRACESTATE_SPLIT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[ \t]*,[ \t]*").expect("valid regex"));

/// Whether `value` may be projected as the correlation id `key` (ASCII; a
/// session id is 8-128 of `[A-Za-z0-9_-]`, anything else at most 256 bytes
/// of `[A-Za-z0-9._:@/-]` starting alphanumeric).
pub(crate) fn is_valid_correlation_id(key: &str, value: &str) -> bool {
    if !value.is_ascii() {
        return false;
    }
    if key == SESSION_ID_BAGGAGE_KEY {
        return SESSION_ID.is_match(value);
    }
    value.len() <= 256 && OPAQUE_ID.is_match(value)
}

/// Whether a `protocol_version` is safe to put on the connection span.
pub(crate) fn is_safe_protocol_version(value: &str) -> bool {
    PROTOCOL_VERSION.is_match(value)
}

/// Header values decoded as Latin-1, as the ASGI server does.
fn header_values(headers: &HeaderMap, name: &str) -> Vec<String> {
    headers
        .get_all(name)
        .iter()
        .map(|v| v.as_bytes().iter().map(|&b| char::from(b)).collect())
        .collect()
}

/// A baggage value: valid percent escapes only, decoding to UTF-8.
fn decode_baggage_value(value: &str) -> Option<String> {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = bytes.get(i + 1..i + 3)?;
            let text = std::str::from_utf8(hex).ok()?;
            if !hex.iter().all(u8::is_ascii_hexdigit) {
                return None;
            }
            out.push(u8::from_str_radix(text, 16).ok()?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

/// The allow-listed baggage members, and whether anything was invalid
/// (upstream `_copy_voice_baggage`).
pub(crate) fn voice_baggage(headers: &HeaderMap) -> (Vec<(String, String)>, bool) {
    let raw = header_values(headers, "baggage");
    if raw.is_empty() {
        return (Vec::new(), false);
    }
    let header = raw.join(",");
    if header.chars().count() > 8192 {
        return (Vec::new(), true);
    }
    let members: Vec<&str> = header.split(',').collect();
    if members.len() > 180 {
        return (Vec::new(), true);
    }
    let mut approved: Vec<(&str, Vec<&str>)> = Vec::new();
    let mut invalid = false;
    for member in members {
        let candidate = member.trim();
        let Some((key, value)) = candidate.split_once('=') else {
            invalid |= BAGGAGE_KEYS.contains(&candidate);
            continue;
        };
        if !BAGGAGE_KEYS.contains(&key) {
            continue;
        }
        match approved.iter_mut().find(|(k, _)| *k == key) {
            Some((_, values)) => values.push(value),
            None => approved.push((key, vec![value])),
        }
    }
    let mut kept = Vec::new();
    for (key, values) in approved {
        let [value] = values.as_slice() else {
            invalid = true;
            continue;
        };
        if value.contains(';') {
            invalid = true;
            continue;
        }
        match decode_baggage_value(value) {
            Some(decoded) if is_valid_correlation_id(key, &decoded) => {
                kept.push((key.to_string(), decoded))
            }
            _ => invalid = true,
        }
    }
    (kept, invalid)
}

/// Whether the `tracestate` headers are valid W3C (upstream
/// `_has_valid_tracestate`): at most 512 bytes and 32 unique members.
pub(crate) fn has_valid_tracestate(headers: &HeaderMap) -> bool {
    let values = header_values(headers, "tracestate");
    if values.is_empty() {
        return true;
    }
    let header = values.join(",");
    if header.chars().count() > 512 {
        return false;
    }
    let members: Vec<&str> = TRACESTATE_SPLIT.split(&header).collect();
    if members.len() > 32 {
        return false;
    }
    let mut keys = std::collections::HashSet::new();
    for member in members {
        let Some(captures) = TRACESTATE_MEMBER.captures(member) else {
            return false;
        };
        if member.is_empty() || !keys.insert(captures[1].to_string()) {
            return false;
        }
    }
    true
}

/// What the upgrade request carries for the connection's trace.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct Propagation {
    /// The `traceparent`, when exactly one was sent.
    pub(crate) traceparent: Option<String>,
    /// The joined `tracestate`, when valid.
    pub(crate) tracestate: Option<String>,
    /// The kept baggage members, `x_request_id` included.
    pub(crate) baggage: Vec<(String, String)>,
    /// The propagation failure to report (`missing` or `invalid`).
    pub(crate) failure: Option<&'static str>,
}

/// Read the trace context of an upgrade request (upstream
/// `_extract_voice_websocket_context`, minus the OpenTelemetry extraction,
/// which [`connection_span`] does).
pub(crate) fn propagation(headers: &HeaderMap) -> Propagation {
    let mut failure = None;
    let traceparents = header_values(headers, "traceparent");
    match traceparents.len() {
        0 => failure = Some("missing"),
        1 => {}
        _ => failure = Some("invalid"),
    }
    let traceparent = match traceparents.as_slice() {
        [one] => Some(one.clone()),
        _ => None,
    };
    let tracestate = if has_valid_tracestate(headers) {
        let values = header_values(headers, "tracestate");
        (!values.is_empty()).then(|| values.join(","))
    } else {
        failure = Some("invalid");
        None
    };
    if let Some(tp) = &traceparent {
        if crate::host::parse_traceparent(tp).is_none() {
            failure = Some("invalid");
        }
    }
    let (mut baggage, invalid_baggage) = voice_baggage(headers);
    if invalid_baggage {
        failure = Some("invalid");
    }
    let request_ids = header_values(headers, crate::scope::headers::REQUEST_ID);
    match request_ids.as_slice() {
        [] => {}
        [id] if !id.is_empty() && is_valid_correlation_id("x_request_id", id) => {
            baggage.push(("x_request_id".into(), id.clone()));
        }
        _ => failure = Some("invalid"),
    }
    Propagation {
        traceparent,
        tracestate,
        baggage,
        failure,
    }
}

/// The `agentserver.connection` span of one connection.
pub(crate) fn connection_span(headers: &HeaderMap, session_id: &str) -> tracing::Span {
    let propagation = propagation(headers);
    let span = tracing::info_span!(
        target: "azure.ai.agentserver.invocations.voice",
        parent: None,
        "agentserver.connection",
        otel.kind = "server",
        network.protocol.name = "websocket",
        azure.ai.agentserver.invocations_ws.session_id = tracing::field::Empty,
        azure.ai.agentserver.invocations_ws.reconnect = tracing::field::Empty,
        azure.ai.agentserver.invocations_ws.protocol_version = tracing::field::Empty,
        azure.ai.agentserver.invocations_ws.close_code = tracing::field::Empty,
        bridge.outcome = tracing::field::Empty,
        error.type = tracing::field::Empty,
        otel.status_code = tracing::field::Empty,
        trace_id = tracing::field::Empty,
        parent_span_id = tracing::field::Empty,
    );
    if is_valid_correlation_id(SESSION_ID_BAGGAGE_KEY, session_id) {
        span.record("azure.ai.agentserver.invocations_ws.session_id", session_id);
    }
    #[cfg(not(feature = "otel"))]
    if let Some((trace_id, parent_id)) = propagation
        .traceparent
        .as_deref()
        .and_then(crate::host::parse_traceparent)
    {
        span.record("trace_id", trace_id.as_str());
        span.record("parent_span_id", parent_id.as_str());
    }
    #[cfg(feature = "otel")]
    {
        use opentelemetry::baggage::BaggageExt as _;
        use opentelemetry::propagation::TextMapPropagator as _;
        use opentelemetry::KeyValue;
        use tracing_opentelemetry::OpenTelemetrySpanExt as _;

        struct Carrier<'a>(&'a Propagation);
        impl opentelemetry::propagation::Extractor for Carrier<'_> {
            fn get(&self, key: &str) -> Option<&str> {
                match key {
                    "traceparent" => self.0.traceparent.as_deref(),
                    "tracestate" => self.0.tracestate.as_deref(),
                    _ => None,
                }
            }
            fn keys(&self) -> Vec<&str> {
                vec!["traceparent", "tracestate"]
            }
        }
        let parent = opentelemetry_sdk::propagation::TraceContextPropagator::new()
            .extract_with_context(&opentelemetry::Context::new(), &Carrier(&propagation));
        let parent = parent.with_baggage(
            propagation
                .baggage
                .iter()
                .map(|(k, v)| KeyValue::new(k.clone(), v.clone())),
        );
        let _ = span.set_parent(parent);
    }
    if let Some(failure) = propagation.failure {
        record_propagation_failure(failure);
    }
    span
}

/// A `voice.callback` span for `event_type` under the connection span.
pub(crate) fn callback_span(connection: &tracing::Span, event_type: &str) -> tracing::Span {
    tracing::info_span!(
        target: "azure.ai.agentserver.invocations.voice",
        parent: connection,
        "voice.callback",
        otel.kind = "internal",
        voice.event.type = event_type,
        error.type = tracing::field::Empty,
        otel.status_code = tracing::field::Empty,
    )
}

/// Mark a span failed with a content-free error type.
pub(crate) fn record_error(span: &tracing::Span, error_type: &str) {
    span.record("error.type", error_type);
    span.record("otel.status_code", "ERROR");
}

/// Record a connection's terminal facts on its span, and its duration.
pub(crate) fn complete_connection(
    span: &tracing::Span,
    outcome: &str,
    close_code: u16,
    duration: Duration,
) {
    span.record(
        "azure.ai.agentserver.invocations_ws.close_code",
        i64::from(close_code),
    );
    span.record("bridge.outcome", outcome);
    let error_type = (!matches!(outcome, "completed" | "cancelled")).then_some(outcome);
    if let Some(error_type) = error_type {
        record_error(span, error_type);
    }
    #[cfg(feature = "otel")]
    {
        let mut attributes = vec![opentelemetry::KeyValue::new(
            "bridge.outcome",
            outcome.to_string(),
        )];
        if let Some(error_type) = error_type {
            attributes.push(opentelemetry::KeyValue::new(
                "error.type",
                error_type.to_string(),
            ));
        }
        INSTRUMENTS
            .connection_duration
            .record(duration.as_secs_f64(), &attributes);
    }
    #[cfg(not(feature = "otel"))]
    let _ = duration;
}

/// Record one target-turn duration (`gen_ai.invoke_agent.duration`).
pub(crate) fn record_target_duration(
    duration: Duration,
    origin: &str,
    outcome: &str,
    error_type: Option<&str>,
) {
    #[cfg(feature = "otel")]
    {
        use opentelemetry::KeyValue;
        let mut attributes = vec![
            KeyValue::new("turn.origin", origin.to_string()),
            KeyValue::new("bridge.outcome", outcome.to_string()),
        ];
        if let Some(error_type) = error_type {
            attributes.push(KeyValue::new("error.type", error_type.to_string()));
        }
        INSTRUMENTS
            .target_duration
            .record(duration.as_secs_f64(), &attributes);
    }
    #[cfg(not(feature = "otel"))]
    let _ = (duration, origin, outcome, error_type);
}

fn record_propagation_failure(error_type: &'static str) {
    #[cfg(feature = "otel")]
    INSTRUMENTS.propagation_failures.add(
        1,
        &[
            opentelemetry::KeyValue::new(
                "azure.ai.agentserver.trace_context.propagation.hop",
                "hosted_agents_to_agentserver",
            ),
            opentelemetry::KeyValue::new("error.type", error_type),
        ],
    );
    #[cfg(not(feature = "otel"))]
    let _ = error_type;
}

#[cfg(feature = "otel")]
struct Instruments {
    connection_duration: opentelemetry::metrics::Histogram<f64>,
    target_duration: opentelemetry::metrics::Histogram<f64>,
    propagation_failures: opentelemetry::metrics::Counter<u64>,
}

#[cfg(feature = "otel")]
static INSTRUMENTS: LazyLock<Instruments> = LazyLock::new(|| {
    let meter = opentelemetry::global::meter_with_scope(
        opentelemetry::InstrumentationScope::builder(SCOPE_NAME)
            .with_version(env!("CARGO_PKG_VERSION"))
            .with_schema_url(SCHEMA_URL)
            .build(),
    );
    Instruments {
        connection_duration: meter
            .f64_histogram("azure.ai.agentserver.voice.connection.duration")
            .with_unit("s")
            .with_boundaries(vec![
                1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0, 7200.0,
            ])
            .build(),
        target_duration: meter
            .f64_histogram("gen_ai.invoke_agent.duration")
            .with_unit("s")
            .with_boundaries(vec![
                0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2, 102.4, 204.8, 409.6,
            ])
            .build(),
        propagation_failures: meter
            .u64_counter("azure.ai.agentserver.trace_context.propagation_failures")
            .build(),
    }
});

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.append(*name, HeaderValue::from_str(value).unwrap());
        }
        map
    }

    #[test]
    fn baggage_keeps_only_valid_allow_listed_members() {
        let (kept, invalid) = voice_baggage(&headers(&[(
            "baggage",
            "azure.ai.agentserver.session_id=session_123,user.id=u%40example,other=x",
        )]));
        assert_eq!(
            kept,
            vec![
                (
                    "azure.ai.agentserver.session_id".into(),
                    "session_123".into()
                ),
                ("user.id".into(), "u@example".into()),
            ]
        );
        assert!(!invalid);
        // A duplicate, a property, a short session id, a bad escape.
        for bad in [
            "user.id=a,user.id=b",
            "user.id=a;prop=1",
            "azure.ai.agentserver.session_id=short",
            "user.id=%zz",
            "user.id",
        ] {
            let (kept, invalid) = voice_baggage(&headers(&[("baggage", bad)]));
            assert!(kept.is_empty() && invalid, "{bad}");
        }
        // Repeated headers are joined.
        let (kept, _) = voice_baggage(&headers(&[
            ("baggage", "user.id=a"),
            ("baggage", "gen_ai.agent.id=b"),
        ]));
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn tracestate_is_validated() {
        assert!(has_valid_tracestate(&headers(&[(
            "tracestate",
            "a=1,b@c=2"
        )])));
        assert!(has_valid_tracestate(&headers(&[])));
        assert!(!has_valid_tracestate(&headers(&[(
            "tracestate",
            "a=1,a=2"
        )])));
        assert!(!has_valid_tracestate(&headers(&[("tracestate", "A=1")])));
        assert!(!has_valid_tracestate(&headers(&[(
            "tracestate",
            "a=1,,b=2"
        )])));
        let long = format!("a={}", "x".repeat(255));
        let members: Vec<String> = (0..3)
            .map(|i| format!("k{i}={}", "x".repeat(200)))
            .collect();
        assert!(has_valid_tracestate(&headers(&[("tracestate", &long)])));
        assert!(!has_valid_tracestate(&headers(&[(
            "tracestate",
            &members.join(",")
        )])));
    }

    #[test]
    fn propagation_classifies_failures() {
        let tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
        assert_eq!(propagation(&headers(&[])).failure, Some("missing"));
        assert_eq!(propagation(&headers(&[("traceparent", tp)])).failure, None);
        let twice = propagation(&headers(&[("traceparent", tp), ("traceparent", tp)]));
        assert_eq!((twice.failure, twice.traceparent), (Some("invalid"), None));
        let bad_state = propagation(&headers(&[("traceparent", tp), ("tracestate", "A=1")]));
        assert_eq!(
            (bad_state.failure, bad_state.tracestate),
            (Some("invalid"), None)
        );
        let request = propagation(&headers(&[("traceparent", tp), ("x-request-id", "req-1")]));
        assert_eq!(
            request.baggage,
            vec![("x_request_id".into(), "req-1".into())]
        );
        let bad_request = propagation(&headers(&[("traceparent", tp), ("x-request-id", "-bad")]));
        assert_eq!(bad_request.failure, Some("invalid"));
        assert!(bad_request.baggage.is_empty());
    }

    #[test]
    fn correlation_ids_are_bounded() {
        assert!(is_valid_correlation_id(
            SESSION_ID_BAGGAGE_KEY,
            "session_safe_123"
        ));
        assert!(!is_valid_correlation_id(SESSION_ID_BAGGAGE_KEY, "short"));
        assert!(is_valid_correlation_id("user.id", "a:b@c/d"));
        assert!(!is_valid_correlation_id("user.id", &"a".repeat(257)));
        assert!(is_safe_protocol_version("1.0"));
        assert!(!is_safe_protocol_version("private-protocol-sentinel"));
    }
}
