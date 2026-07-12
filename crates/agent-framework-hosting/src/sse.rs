//! Shared Server-Sent-Events framing.

use std::convert::Infallible;

use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use futures::{Stream, StreamExt};
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

/// Live variant of [`sse_response`]: frame a **stream** of JSON event objects as
/// an SSE response, appending a final `data: [DONE]` once the stream ends. Used
/// by the streaming devui / OpenAI-compat paths that consume `SupportsAgentRun::run_stream`
/// incrementally.
pub(crate) fn sse_response_stream<S>(events: S) -> Response
where
    S: Stream<Item = Value> + Send + 'static,
{
    let stream = events
        .map(|v| {
            Ok::<_, Infallible>(
                Event::default().data(serde_json::to_string(&v).unwrap_or_default()),
            )
        })
        .chain(futures::stream::once(async {
            Ok::<_, Infallible>(Event::default().data("[DONE]"))
        }));
    Sse::new(stream).into_response()
}

/// Live variant of [`sse_events`]: frame a **stream** of JSON event objects as
/// an SSE response with no terminal sentinel (AG-UI). Used by the AG-UI router
/// to emit events as `SupportsAgentRun::run_stream` produces them.
pub(crate) fn sse_events_stream<S>(events: S) -> Response
where
    S: Stream<Item = Value> + Send + 'static,
{
    let stream = events.map(|v| {
        Ok::<_, Infallible>(Event::default().data(serde_json::to_string(&v).unwrap_or_default()))
    });
    Sse::new(stream).into_response()
}
