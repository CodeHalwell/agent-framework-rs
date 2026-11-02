//! The application-owned trace for one target-agent decision (upstream
//! `voice/_turn.py`).

use std::future::Future;
use std::time::Instant;

use tracing::Instrument;

use super::codec::validate_prefixed_identifier;
use super::telemetry;
use super::VoiceError;

const MAX_COUNT: u64 = i64::MAX as u64;

/// The application-declared origin of a target-agent decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetTurnOrigin {
    /// A user message.
    User,
    /// A `user.no_input` turn.
    NoInput,
    /// Proactive output (no inputs).
    Proactive,
    /// Recovery work.
    Recovery,
    /// Anything else.
    Other,
}

impl TargetTurnOrigin {
    /// The telemetry value.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::NoInput => "no_input",
            Self::Proactive => "proactive",
            Self::Recovery => "recovery",
            Self::Other => "other",
        }
    }

    /// Parse an open-enum value: an unknown value is `Other`, an empty one
    /// an error (as upstream).
    pub fn parse(value: &str) -> Result<Self, VoiceError> {
        if value.trim().is_empty() {
            return Err(VoiceError::Invalid("origin must not be empty".into()));
        }
        Ok(match value {
            "user" => Self::User,
            "no_input" => Self::NoInput,
            "proactive" => Self::Proactive,
            "recovery" => Self::Recovery,
            _ => Self::Other,
        })
    }
}

/// The terminal outcome of a target-agent decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetTurnOutcome {
    /// A response with at least one output item.
    Response,
    /// `response.none`.
    None,
    /// The Bridge timed it out.
    Timeout,
    /// The agent failed.
    Error,
    /// Cancelled (barge-in, self-cancel).
    Cancelled,
    /// The call ended.
    EndCall,
    /// The transport failed.
    TransportError,
    /// Abandoned before completion.
    Abandoned,
    /// Anything else.
    Other,
}

impl TargetTurnOutcome {
    /// The telemetry value.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Response => "response",
            Self::None => "none",
            Self::Timeout => "timeout",
            Self::Error => "error",
            Self::Cancelled => "cancelled",
            Self::EndCall => "end_call",
            Self::TransportError => "transport_error",
            Self::Abandoned => "abandoned",
            Self::Other => "other",
        }
    }

    /// Parse an open-enum value: an unknown value is `Other`, an empty one
    /// an error (as upstream).
    pub fn parse(value: &str) -> Result<Self, VoiceError> {
        if value.trim().is_empty() {
            return Err(VoiceError::Invalid("outcome must not be empty".into()));
        }
        Ok(match value {
            "response" => Self::Response,
            "none" => Self::None,
            "timeout" => Self::Timeout,
            "error" => Self::Error,
            "cancelled" => Self::Cancelled,
            "end_call" => Self::EndCall,
            "transport_error" => Self::TransportError,
            "abandoned" => Self::Abandoned,
            _ => Self::Other,
        })
    }

    fn is_error(self) -> bool {
        matches!(
            self,
            Self::Timeout | Self::Error | Self::TransportError | Self::Abandoned
        )
    }
}

fn check_count(value: u64, name: &str, minimum: u64) -> Result<u64, VoiceError> {
    if value < minimum || value > MAX_COUNT {
        return Err(VoiceError::Invalid(format!(
            "{name} must be between {minimum} and {MAX_COUNT}"
        )));
    }
    Ok(value)
}

/// A response id safe to project into telemetry (`r_` and 32 lowercase hex
/// digits, as [`new_response_id`](super::new_response_id) makes).
fn is_safe_telemetry_response_id(value: &str) -> bool {
    value.strip_prefix("r_").is_some_and(|hex| {
        hex.len() == 32 && hex.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    })
}

/// The application-owned trace lifetime of one target-agent decision: an
/// `invoke_agent` span (`gen_ai.operation.name`, `turn.origin`,
/// `bridge.input.count`) under the connection span. Run the decision's
/// work in it once ([`activate`](Self::activate)), then
/// [`complete`](Self::complete) it with the terminal facts.
#[derive(Debug)]
pub struct TargetTurn {
    span: Option<tracing::Span>,
    origin: TargetTurnOrigin,
    started: Instant,
    activated: bool,
    completed: bool,
}

impl TargetTurn {
    pub(crate) fn create(
        connection_span: &tracing::Span,
        origin: TargetTurnOrigin,
        input_count: u64,
    ) -> Result<Self, VoiceError> {
        let minimum = u64::from(!matches!(
            origin,
            TargetTurnOrigin::Proactive | TargetTurnOrigin::Other
        ));
        let input_count = check_count(input_count, "input_count", minimum)?;
        if origin == TargetTurnOrigin::Proactive && input_count != 0 {
            return Err(VoiceError::Invalid(
                "input_count must be zero for proactive target turns".into(),
            ));
        }
        let span = tracing::info_span!(
            target: "azure.ai.agentserver.invocations.voice",
            parent: connection_span,
            "invoke_agent",
            otel.kind = "internal",
            gen_ai.operation.name = "invoke_agent",
            turn.origin = origin.as_str(),
            bridge.input.count = input_count,
            bridge.outcome = tracing::field::Empty,
            bridge.output.item_count = tracing::field::Empty,
            gen_ai.response.id = tracing::field::Empty,
            error.type = tracing::field::Empty,
            otel.status_code = tracing::field::Empty,
        );
        Ok(Self {
            span: Some(span),
            origin,
            started: Instant::now(),
            activated: false,
            completed: false,
        })
    }

    #[cfg(feature = "otel")]
    pub(crate) fn link(
        &self,
        trigger: opentelemetry::trace::SpanContext,
    ) -> Result<(), VoiceError> {
        use opentelemetry::trace::{SpanContext, TraceState};
        use tracing_opentelemetry::OpenTelemetrySpanExt;
        if !trigger.is_valid() {
            return Err(VoiceError::Invalid(
                "trigger_context must contain a valid trace ID and span ID".into(),
            ));
        }
        let safe = SpanContext::new(
            trigger.trace_id(),
            trigger.span_id(),
            trigger.trace_flags(),
            trigger.is_remote(),
            TraceState::default(),
        );
        if let Some(span) = &self.span {
            span.add_link(safe);
        }
        Ok(())
    }

    /// Whether the first valid completion has committed.
    pub fn is_completed(&self) -> bool {
        self.completed
    }

    /// The `invoke_agent` span (until completion), for work the application
    /// runs outside [`activate`](Self::activate).
    pub fn span(&self) -> Option<&tracing::Span> {
        self.span.as_ref()
    }

    /// Run `work` with this turn's span current (its single activation
    /// scope; upstream `activate()`). Await every task that creates target
    /// descendants inside it. A turn cannot complete while it is active.
    pub async fn activate<F: Future>(&mut self, work: F) -> Result<F::Output, VoiceError> {
        if self.completed {
            return Err(VoiceError::Invalid(
                "Target turn has already completed".into(),
            ));
        }
        if self.activated {
            return Err(VoiceError::Invalid(
                "Target turn has already been activated".into(),
            ));
        }
        self.activated = true;
        let span = self.span.clone().unwrap_or_else(tracing::Span::none);
        Ok(work.instrument(span).await)
    }

    /// Complete the turn with application-owned terminal facts (only the
    /// first completion counts):
    ///
    /// - [`TargetTurnOutcome::Response`] requires `response_id` and an
    ///   `output_item_count` of at least 1;
    /// - [`TargetTurnOutcome::None`] requires no `response_id` and an
    ///   `output_item_count` of 0;
    /// - other outcomes allow both to be absent, but a positive
    ///   `output_item_count` requires `response_id`.
    pub fn complete(
        &mut self,
        outcome: TargetTurnOutcome,
        response_id: Option<&str>,
        output_item_count: Option<u64>,
    ) -> Result<(), VoiceError> {
        if self.completed {
            return Ok(());
        }
        if let Some(response_id) = response_id {
            validate_prefixed_identifier(response_id, "response_id", "r_")
                .map_err(VoiceError::Invalid)?;
        }
        if let Some(count) = output_item_count {
            check_count(count, "output_item_count", 0)?;
        }
        match outcome {
            TargetTurnOutcome::Response
                if response_id.is_none() || output_item_count.is_none_or(|c| c < 1) =>
            {
                return Err(VoiceError::Invalid(
                    "response outcome requires a response_id and at least one output item".into(),
                ));
            }
            TargetTurnOutcome::None if response_id.is_some() || output_item_count != Some(0) => {
                return Err(VoiceError::Invalid(
                    "none outcome requires no response_id and zero output items".into(),
                ));
            }
            TargetTurnOutcome::Response | TargetTurnOutcome::None => {}
            _ if output_item_count.is_some_and(|c| c > 0) && response_id.is_none() => {
                return Err(VoiceError::Invalid(
                    "positive output_item_count requires a response_id".into(),
                ));
            }
            _ => {}
        }
        self.completed = true;
        let error_type = outcome.is_error().then(|| outcome.as_str());
        if let Some(span) = self.span.take() {
            span.record("bridge.outcome", outcome.as_str());
            if let Some(count) = output_item_count {
                span.record("bridge.output.item_count", count);
            }
            if let Some(id) = response_id.filter(|id| is_safe_telemetry_response_id(id)) {
                span.record("gen_ai.response.id", id);
            }
            if let Some(error_type) = error_type {
                span.record("error.type", error_type);
                span.record("otel.status_code", "ERROR");
            }
            // Dropping the span ends it.
        }
        telemetry::record_target_duration(
            self.started.elapsed(),
            self.origin.as_str(),
            outcome.as_str(),
            error_type,
        );
        Ok(())
    }
}
