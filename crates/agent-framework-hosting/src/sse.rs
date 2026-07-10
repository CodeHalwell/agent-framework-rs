//! Shared Server-Sent-Events framing.

use std::convert::Infallible;

use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use serde_json::Value;

/// Frame a list of JSON event objects as an SSE response, appending a final
/// `data: [DONE]` line (the OpenAI streaming terminator).
///
/// Each event is serialized to single-line JSON (no embedded newlines), so the
/// SSE `data:` framing stays valid.
pub(crate) fn sse_response(events: Vec<Value>) -> Response {
    let mut sse_events: Vec<Result<Event, Infallible>> = events
        .into_iter()
        .map(|v| Ok(Event::default().data(serde_json::to_string(&v).unwrap_or_default())))
        .collect();
    sse_events.push(Ok(Event::default().data("[DONE]")));
    Sse::new(futures::stream::iter(sse_events)).into_response()
}
