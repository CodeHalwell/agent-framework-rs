//! Shared Server-Sent-Events framing.

use std::convert::Infallible;

use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use futures::{Stream, StreamExt};
use serde_json::Value;
use tokio::sync::mpsc::Sender;
use tokio_stream::wrappers::ReceiverStream;

/// Capacity of the bounded channel backing a live SSE producer.
///
/// A **bounded** channel is what gives streaming endpoints backpressure and
/// disconnect-driven cancellation: when a slow client can't keep up, the buffer
/// fills and the producer's `send().await` suspends (memory stays bounded); when
/// a client disconnects, the receiver is dropped and the producer's next
/// `send().await` fails (or its `closed()` watcher fires), so the producer task
/// exits and drops the underlying agent run instead of continuing to burn tokens
/// and invoke tools for output nobody will read.
pub(crate) const SSE_CHANNEL_CAPACITY: usize = 32;

/// Create the bounded channel + receiver-stream pair for a live SSE endpoint.
/// The [`Sender`] is used by the producer task (awaiting each send for
/// backpressure); the [`ReceiverStream`] is handed to one of the `*_stream`
/// framing helpers below and drives the HTTP response body.
pub(crate) fn bounded_sse_channel() -> (Sender<Value>, ReceiverStream<Value>) {
    let (tx, rx) = tokio::sync::mpsc::channel(SSE_CHANNEL_CAPACITY);
    (tx, ReceiverStream::new(rx))
}

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
