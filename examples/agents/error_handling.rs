//! Error handling: the granular `Error` variants a provider client raises,
//! which of them the default retry policy treats as transient, and the one
//! error a tool can return that stops a run outright.
//!
//! Three things are shown, all offline against scripted clients:
//!
//! 1. **Matching the variants.** `Error::ServiceStatus` carries the numeric
//!    HTTP status and any server-advised `Retry-After`;
//!    `ServiceInvalidAuth` / `ServiceInvalidRequest` / `ServiceContentFilter`
//!    exist so a caller can tell "your key is wrong" from "your request was
//!    malformed" from "the moderation layer refused this" without parsing a
//!    message string.
//! 2. **Retry classification.** Each error is sent through a
//!    `RetryingChatClient` that counts attempts, so you can *see* which ones
//!    the default policy retries. A custom `RetryOn::predicate` then
//!    overrides that rule.
//! 3. **Tool errors vs. fail-closed.** An ordinary tool error is absorbed
//!    into a `FunctionResultContent` and handed back to the model, which
//!    keeps looping. `Error::MiddlewareFailure` is the single exception: it
//!    propagates and ends the run, which is what a guardrail needs.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example error_handling
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use agent_framework::prelude::*;
use agent_framework::types::FunctionArguments;
use async_trait::async_trait;
use serde_json::json;

/// A client that always fails with a fixed error, counting how many times it
/// was called -- which is how the retry layer's classification becomes
/// observable from outside (`RetryOn::should_retry` is private by design).
struct AlwaysFails {
    make_error: Box<dyn Fn() -> Error + Send + Sync>,
    attempts: Arc<AtomicUsize>,
}

impl AlwaysFails {
    fn new(make_error: impl Fn() -> Error + Send + Sync + 'static) -> Self {
        Self {
            make_error: Box::new(make_error),
            attempts: Arc::new(AtomicUsize::new(0)),
        }
    }
}

#[async_trait]
impl ChatClient for AlwaysFails {
    async fn get_response(&self, _m: Vec<Message>, _o: ChatOptions) -> Result<ChatResponse> {
        self.attempts.fetch_add(1, Ordering::SeqCst);
        Err((self.make_error)())
    }

    async fn get_streaming_response(&self, _m: Vec<Message>, _o: ChatOptions) -> Result<ChatStream> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

/// A one-line, human-readable label for an error variant.
fn classify(err: &Error) -> String {
    match err {
        Error::ServiceStatus {
            status,
            retry_after,
            ..
        } => format!("ServiceStatus  status={status} retry_after={retry_after:?}"),
        Error::ServiceInvalidAuth { .. } => "ServiceInvalidAuth    (bad/missing credentials)".into(),
        Error::ServiceInvalidRequest { .. } => {
            "ServiceInvalidRequest (malformed request)".into()
        }
        Error::ServiceContentFilter { .. } => {
            "ServiceContentFilter  (moderation refusal)".into()
        }
        Error::Service(msg) => format!("Service               (transport-ish) -- {msg}"),
        other => format!("{other:?}"),
    }
}

/// Run `error` through a retrying client and report how many attempts it took.
/// Delays are set to near-zero and jitter disabled so the example finishes
/// instantly and prints the same numbers every time.
async fn attempts_for(
    label: &str,
    make_error: impl Fn() -> Error + Send + Sync + 'static,
    retry_on: RetryOn,
) {
    let inner = AlwaysFails::new(make_error);
    let attempts = inner.attempts.clone();
    let policy = RetryPolicy {
        max_retries: 2,
        initial_delay: Duration::from_millis(1),
        jitter: 0.0,
        retry_on,
        ..RetryPolicy::default()
    };
    let client = RetryingChatClient::new(inner).with_policy(policy);

    let err = client
        .get_response(vec![Message::user("hi")], ChatOptions::new())
        .await
        .expect_err("this client always fails");

    let n = attempts.load(Ordering::SeqCst);
    let verdict = if n > 1 { "RETRIED" } else { "not retried" };
    println!("  {label:<22} {n} attempt(s)  {verdict:<11} -- {}", classify(&err));
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("== 1. the granular service-error variants ==\n");
    for err in [
        Error::service_status(429, "rate limit exceeded", Some(1.5)),
        Error::service_status(503, "upstream unavailable", None),
        Error::service_invalid_auth("401 invalid api key"),
        Error::service_invalid_request("400 unknown parameter 'temperatur'"),
        Error::service_content_filter("response blocked by the content filter"),
        Error::service("request failed: connection reset by peer"),
    ] {
        println!("  {}", classify(&err));
    }

    println!("\n== 2. which of them `RetryOn::Default` retries ==\n");
    println!("  (max_retries = 2, so a retried error shows 3 attempts)\n");

    attempts_for(
        "429 rate limit",
        || Error::service_status(429, "rate limit exceeded", Some(0.0)),
        RetryOn::Default,
    )
    .await;
    attempts_for(
        "503 unavailable",
        || Error::service_status(503, "upstream unavailable", None),
        RetryOn::Default,
    )
    .await;
    attempts_for(
        "connection reset",
        || Error::service("request failed: connection reset by peer"),
        RetryOn::Default,
    )
    .await;
    attempts_for(
        "401 bad key",
        || Error::service_invalid_auth("401 invalid api key"),
        RetryOn::Default,
    )
    .await;
    attempts_for(
        "400 bad request",
        || Error::service_invalid_request("400 unknown parameter"),
        RetryOn::Default,
    )
    .await;
    attempts_for(
        "content filter",
        || Error::service_content_filter("blocked"),
        RetryOn::Default,
    )
    .await;

    println!(
        "\n  The rule: transient statuses (408/429/5xx) and transport failures are\n  \
         retried; auth failures, malformed requests, and content-filter refusals\n  \
         are not -- repeating them unchanged would just repeat the rejection."
    );

    println!("\n== 3. overriding the rule with a custom predicate ==\n");
    // A stricter policy: retry 5xx only, never 429 (e.g. because this
    // deployment would rather shed load than queue behind a rate limit).
    let only_5xx = || {
        RetryOn::predicate(|err| {
            matches!(err, Error::ServiceStatus { status, .. } if *status >= 500)
        })
    };
    attempts_for(
        "429 (custom rule)",
        || Error::service_status(429, "rate limit exceeded", Some(0.0)),
        only_5xx(),
    )
    .await;
    attempts_for(
        "503 (custom rule)",
        || Error::service_status(503, "upstream unavailable", None),
        only_5xx(),
    )
    .await;

    println!("\n== 4. tool errors: absorbed vs. fail-closed ==\n");
    tool_error_handling().await?;

    Ok(())
}

/// A canned model that calls whichever tool it is asked to, once, then
/// summarises whatever result came back.
#[derive(Clone)]
struct ToolCallingClient {
    tool_name: &'static str,
}

#[async_trait]
impl ChatClient for ToolCallingClient {
    async fn get_response(&self, messages: Vec<Message>, _o: ChatOptions) -> Result<ChatResponse> {
        // If a tool result is already in the transcript, the loop has come
        // back round: produce a final answer instead of calling again.
        let has_result = messages
            .iter()
            .any(|m| m.contents.iter().any(|c| matches!(c, Content::FunctionResult(_))));
        if has_result {
            let detail = messages
                .iter()
                .flat_map(|m| m.contents.iter())
                .find_map(|c| match c {
                    Content::FunctionResult(r) => Some(format!("{:?}", r.exception)),
                    _ => None,
                })
                .unwrap_or_default();
            return Ok(ChatResponse::from_text(format!(
                "the tool failed and I was told about it: exception={detail}"
            )));
        }
        Ok(ChatResponse {
            messages: vec![Message::with_contents(
                Role::assistant(),
                vec![Content::FunctionCall(FunctionCallContent::new(
                    "call-1",
                    self.tool_name,
                    Some(FunctionArguments::Raw(json!({}).to_string())),
                ))],
            )],
            finish_reason: Some(FinishReason::tool_calls()),
            ..Default::default()
        })
    }

    async fn get_streaming_response(&self, _m: Vec<Message>, _o: ChatOptions) -> Result<ChatStream> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

async fn tool_error_handling() -> Result<()> {
    // An ordinary failing tool: `Error::Tool` is absorbed into the function
    // result and handed back to the model, which gets to react to it.
    let flaky = FunctionTool::new(
        "flaky",
        "A tool that always fails in the ordinary way.",
        json!({ "type": "object", "properties": {} }),
        |_args| async move { Err::<serde_json::Value, _>(Error::Tool("upstream API is down".into())) },
    )
    .into_definition();

    // A guardrail that refuses: `Error::MiddlewareFailure` is the one error
    // the function-invocation loop propagates instead of absorbing, so the
    // run fails closed rather than letting the model try again.
    let refused = FunctionTool::new(
        "refused",
        "A tool guarded by a policy check that refuses the call.",
        json!({ "type": "object", "properties": {} }),
        |_args| async move {
            Err::<serde_json::Value, _>(Error::MiddlewareFailure(
                "policy: this tool is not permitted for the current principal".into(),
            ))
        },
    )
    .into_definition();

    let agent = Agent::builder(ToolCallingClient { tool_name: "flaky" })
        .name("assistant")
        .tool(flaky)
        .build();
    let response = agent.run_once("go").await?;
    println!("  ordinary tool error -> run SUCCEEDS, model sees the failure:");
    println!("    {}", response.text());

    let agent = Agent::builder(ToolCallingClient {
        tool_name: "refused",
    })
    .name("assistant")
    .tool(refused)
    .build();
    match agent.run_once("go").await {
        Ok(r) => println!("  unexpected success: {}", r.text()),
        Err(e) => println!("\n  MiddlewareFailure    -> run FAILS closed:\n    {e}"),
    }

    Ok(())
}
