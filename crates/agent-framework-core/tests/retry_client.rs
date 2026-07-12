//! `RetryingChatClient` tests: a scripted flaky client exercising attempt
//! counts, `Retry-After` honoring, non-retryable-status short-circuiting,
//! streaming initial-connection retry (and mid-stream propagation), and
//! exhaustion. Timing is asserted instantly via tokio's paused clock
//! (`#[tokio::test(start_paused = true)]`), with jitter disabled for
//! determinism. No network.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_framework_core::prelude::*;
use agent_framework_core::types::ChatResponseUpdate;
use async_trait::async_trait;
use futures::StreamExt;

/// A scripted stream outcome for one `get_streaming_response` call.
enum StreamOutcome {
    /// The stream fails to open (the call returns `Err`).
    OpenError(Error),
    /// The stream opens and yields these items in order.
    Items(Vec<Result<ChatResponseUpdate>>),
}

/// A chat client whose per-call outcomes are fully scripted, counting how many
/// times each entry point was invoked so tests can assert retry attempts.
#[derive(Clone)]
struct FlakyClient {
    resp_calls: Arc<AtomicUsize>,
    responses: Arc<Mutex<VecDeque<Result<ChatResponse>>>>,
    stream_calls: Arc<AtomicUsize>,
    streams: Arc<Mutex<VecDeque<StreamOutcome>>>,
}

impl FlakyClient {
    fn new() -> Self {
        Self {
            resp_calls: Arc::new(AtomicUsize::new(0)),
            responses: Arc::new(Mutex::new(VecDeque::new())),
            stream_calls: Arc::new(AtomicUsize::new(0)),
            streams: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    fn push_response(&self, outcome: Result<ChatResponse>) {
        self.responses.lock().unwrap().push_back(outcome);
    }

    fn push_stream(&self, outcome: StreamOutcome) {
        self.streams.lock().unwrap().push_back(outcome);
    }
}

#[async_trait]
impl ChatClient for FlakyClient {
    async fn get_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        self.resp_calls.fetch_add(1, Ordering::SeqCst);
        self.responses
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| Ok(ChatResponse::from_text("(exhausted script)")))
    }

    async fn get_streaming_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatStream> {
        self.stream_calls.fetch_add(1, Ordering::SeqCst);
        match self.streams.lock().unwrap().pop_front() {
            Some(StreamOutcome::OpenError(e)) => Err(e),
            Some(StreamOutcome::Items(items)) => Ok(futures::stream::iter(items).boxed()),
            None => Ok(futures::stream::empty().boxed()),
        }
    }
}

fn user() -> Vec<Message> {
    vec![Message::user("hi")]
}

fn update(text: &str) -> ChatResponseUpdate {
    ChatResponseUpdate {
        contents: vec![Content::text(text)],
        role: Some(Role::assistant()),
        ..Default::default()
    }
}

/// A default policy with jitter disabled for deterministic virtual-time
/// assertions.
fn deterministic_policy() -> RetryPolicy {
    RetryPolicy::default().jitter(0.0)
}

#[tokio::test(start_paused = true)]
async fn retries_then_succeeds_and_counts_attempts() {
    let client = FlakyClient::new();
    client.push_response(Err(Error::service_status(503, "unavailable", None)));
    client.push_response(Err(Error::service_status(429, "rate limited", None)));
    client.push_response(Ok(ChatResponse::from_text("ok")));
    let calls = client.resp_calls.clone();

    let retrying = RetryingChatClient::new(client).with_policy(deterministic_policy());
    let resp = retrying
        .get_response(user(), ChatOptions::default())
        .await
        .unwrap();

    assert_eq!(resp.text(), "ok");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        3,
        "two retryable failures then success = 3 attempts"
    );
}

#[tokio::test(start_paused = true)]
async fn honors_retry_after_over_computed_backoff() {
    let client = FlakyClient::new();
    // initial_delay is 500ms, but the server asks for 5s: the server wins.
    client.push_response(Err(Error::service_status(429, "slow down", Some(5.0))));
    client.push_response(Ok(ChatResponse::from_text("ok")));

    let retrying = RetryingChatClient::new(client).with_policy(
        deterministic_policy()
            .initial_delay(Duration::from_millis(500))
            .max_delay(Duration::from_secs(60)),
    );

    let start = tokio::time::Instant::now();
    let resp = retrying
        .get_response(user(), ChatOptions::default())
        .await
        .unwrap();
    let elapsed = start.elapsed();

    assert_eq!(resp.text(), "ok");
    assert_eq!(
        elapsed,
        Duration::from_secs(5),
        "waited exactly the server's Retry-After (not the 500ms backoff)"
    );
}

#[tokio::test(start_paused = true)]
async fn retry_after_is_capped_by_max_delay() {
    let client = FlakyClient::new();
    // Server asks for 30s but max_delay is 2s: the cap wins.
    client.push_response(Err(Error::service_status(503, "overloaded", Some(30.0))));
    client.push_response(Ok(ChatResponse::from_text("ok")));

    let retrying = RetryingChatClient::new(client)
        .with_policy(deterministic_policy().max_delay(Duration::from_secs(2)));

    let start = tokio::time::Instant::now();
    retrying
        .get_response(user(), ChatOptions::default())
        .await
        .unwrap();
    assert_eq!(
        start.elapsed(),
        Duration::from_secs(2),
        "capped at max_delay"
    );
}

#[tokio::test(start_paused = true)]
async fn does_not_retry_non_retryable_status() {
    let client = FlakyClient::new();
    client.push_response(Err(Error::service_status(400, "bad request", None)));
    client.push_response(Ok(ChatResponse::from_text("unreached")));
    let calls = client.resp_calls.clone();

    let retrying = RetryingChatClient::new(client).with_policy(deterministic_policy());
    let err = retrying
        .get_response(user(), ChatOptions::default())
        .await
        .unwrap_err();

    assert_eq!(err.status(), Some(400));
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "a 400 is not retryable, so exactly one attempt"
    );
}

/// Auth failures, invalid-request rejections, and content-filter refusals are
/// all definitively non-transient (retrying repeats the exact same
/// rejection), so none of the three new granular variants are retried by
/// [`RetryOn::Default`] — while a `429`/`5xx` [`Error::ServiceStatus`] (the
/// same status family upstream would previously have collapsed these into)
/// is still retried. This is the one behavioral guarantee the whole
/// error-granularity change depends on: adding the variants must not make
/// auth/content-filter errors *more* retryable than the `ServiceStatus` they
/// used to be reported as.
#[tokio::test(start_paused = true)]
async fn does_not_retry_auth_invalid_request_or_content_filter_but_still_retries_429_and_5xx() {
    for (err, label) in [
        (Error::service_invalid_auth("unauthorized"), "auth"),
        (
            Error::service_invalid_request("bad request"),
            "invalid_request",
        ),
        (
            Error::service_content_filter("flagged by content filter"),
            "content_filter",
        ),
    ] {
        let client = FlakyClient::new();
        client.push_response(Err(err));
        client.push_response(Ok(ChatResponse::from_text("unreached")));
        let calls = client.resp_calls.clone();

        let retrying = RetryingChatClient::new(client).with_policy(deterministic_policy());
        let err = retrying
            .get_response(user(), ChatOptions::default())
            .await
            .unwrap_err();

        assert_eq!(err.status(), None, "{label}: not a ServiceStatus");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "{label}: non-transient error must not be retried"
        );
    }

    // Control: the status family these used to be reported under is still
    // retried, so the exclusion above is specific to the new variants rather
    // than a change to the general retry behavior.
    for status in [429, 500, 503] {
        let client = FlakyClient::new();
        client.push_response(Err(Error::service_status(status, "transient", None)));
        client.push_response(Ok(ChatResponse::from_text("ok")));
        let calls = client.resp_calls.clone();

        let retrying = RetryingChatClient::new(client).with_policy(deterministic_policy());
        let resp = retrying
            .get_response(user(), ChatOptions::default())
            .await
            .unwrap();

        assert_eq!(resp.text(), "ok", "status {status}");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "status {status}: still retried once then succeeds"
        );
    }
}

#[tokio::test(start_paused = true)]
async fn retries_transport_service_errors() {
    let client = FlakyClient::new();
    // The provider clients wrap reqwest send failures as `service error:
    // request failed: ...`; the default predicate treats those as transient.
    client.push_response(Err(Error::service("request failed: connection reset")));
    client.push_response(Ok(ChatResponse::from_text("ok")));
    let calls = client.resp_calls.clone();

    let retrying = RetryingChatClient::new(client).with_policy(deterministic_policy());
    let resp = retrying
        .get_response(user(), ChatOptions::default())
        .await
        .unwrap();

    assert_eq!(resp.text(), "ok");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "transport error retried once"
    );
}

#[tokio::test(start_paused = true)]
async fn custom_predicate_overrides_default() {
    let client = FlakyClient::new();
    // A 400 is normally non-retryable; a custom predicate opts into it.
    client.push_response(Err(Error::service_status(400, "retry me", None)));
    client.push_response(Ok(ChatResponse::from_text("ok")));
    let calls = client.resp_calls.clone();

    let retrying = RetryingChatClient::new(client).with_policy(
        deterministic_policy().retry_on(RetryOn::predicate(|e| e.status() == Some(400))),
    );
    let resp = retrying
        .get_response(user(), ChatOptions::default())
        .await
        .unwrap();

    assert_eq!(resp.text(), "ok");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test(start_paused = true)]
async fn exhaustion_returns_the_last_error() {
    let client = FlakyClient::new();
    client.push_response(Err(Error::service_status(500, "first", None)));
    client.push_response(Err(Error::service_status(502, "second", None)));
    client.push_response(Err(Error::service_status(503, "last", None)));
    client.push_response(Ok(ChatResponse::from_text("unreached")));
    let calls = client.resp_calls.clone();

    let retrying =
        RetryingChatClient::new(client).with_policy(RetryPolicy::with_max_retries(2).jitter(0.0));
    let err = retrying
        .get_response(user(), ChatOptions::default())
        .await
        .unwrap_err();

    assert_eq!(
        err.status(),
        Some(503),
        "the final attempt's error propagates"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        3,
        "initial attempt + 2 retries, then give up"
    );
}

#[tokio::test(start_paused = true)]
async fn retries_stream_open_error_then_streams() {
    let client = FlakyClient::new();
    client.push_stream(StreamOutcome::OpenError(Error::service_status(
        503,
        "unavailable",
        None,
    )));
    client.push_stream(StreamOutcome::Items(vec![
        Ok(update("hello")),
        Ok(update(" world")),
    ]));
    let calls = client.stream_calls.clone();

    let retrying = RetryingChatClient::new(client).with_policy(deterministic_policy());
    let stream = retrying
        .get_streaming_response(user(), ChatOptions::default())
        .await
        .unwrap();
    let updates: Vec<ChatResponseUpdate> = stream.map(|r| r.unwrap()).collect().await;

    assert_eq!(calls.load(Ordering::SeqCst), 2, "reconnected once");
    assert_eq!(ChatResponse::from_updates(updates).text(), "hello world");
}

#[tokio::test(start_paused = true)]
async fn retries_stream_first_item_error_then_streams() {
    let client = FlakyClient::new();
    // The stream opens, but its first item is a retryable error before
    // anything is yielded to the consumer: still an initial-connection retry.
    client.push_stream(StreamOutcome::Items(vec![Err(Error::service_status(
        429, "rate", None,
    ))]));
    client.push_stream(StreamOutcome::Items(vec![Ok(update("recovered"))]));
    let calls = client.stream_calls.clone();

    let retrying = RetryingChatClient::new(client).with_policy(deterministic_policy());
    let stream = retrying
        .get_streaming_response(user(), ChatOptions::default())
        .await
        .unwrap();
    let updates: Vec<ChatResponseUpdate> = stream.map(|r| r.unwrap()).collect().await;

    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "first-item error reconnected"
    );
    assert_eq!(ChatResponse::from_updates(updates).text(), "recovered");
}

#[tokio::test(start_paused = true)]
async fn stream_error_after_first_item_propagates() {
    let client = FlakyClient::new();
    // One good item then an error: once updates flow, errors are NOT retried.
    client.push_stream(StreamOutcome::Items(vec![
        Ok(update("partial")),
        Err(Error::service_status(503, "mid-stream", None)),
    ]));
    let calls = client.stream_calls.clone();

    let retrying = RetryingChatClient::new(client).with_policy(deterministic_policy());
    let mut stream = retrying
        .get_streaming_response(user(), ChatOptions::default())
        .await
        .unwrap();

    let first = stream.next().await.expect("a first item");
    assert!(first.is_ok(), "first update flows through");
    let second = stream.next().await.expect("a second item");
    assert!(
        second.is_err(),
        "a mid-stream error propagates instead of reconnecting"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "no reconnection once the first update has been yielded"
    );
}
