//! Isolated span-capture smoke tests for [`ObservableChatClient`] and the
//! `execute_tool` span helpers.
//!
//! This is a dedicated test binary on purpose: `tracing` caches callsite
//! interest globally, so if a span callsite (`chat`, `execute_tool`, ...)
//! were first evaluated under the default no-op subscriber (in another
//! test), it would be cached as disabled and never fire under a thread-local
//! capturing subscriber. Keeping this the only test binary that touches
//! those callsites makes capture deterministic — and within this binary,
//! every test that captures spans runs its capturing subscriber through
//! [`run_captured`], serialized by `SPAN_TEST_MUTEX`, so tests never race to
//! be the "first" observer of a callsite.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use agent_framework_core::observability::{
    attr, record_error, record_tool_arguments, record_tool_result, tool_span, tool_span_ex,
    ObservableChatClient,
};
use agent_framework_core::prelude::*;
use async_trait::async_trait;
use tracing::Instrument;
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::Layer;

/// A chat client that returns a single canned response (optionally
/// overriding the response model id) or, if `fail` is set, an error —
/// echoing usage/finish attributes either way.
#[derive(Clone, Default)]
struct StubClient {
    fail: bool,
    response_model: Option<String>,
}

#[async_trait]
impl ChatClient for StubClient {
    async fn get_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        if self.fail {
            return Err(Error::service("boom"));
        }
        let mut resp = ChatResponse::from_text("hi");
        resp.finish_reason = Some(FinishReason::stop());
        resp.model = self.response_model.clone();
        resp.usage_details = Some(UsageDetails {
            input_token_count: Some(7),
            output_token_count: Some(3),
            total_token_count: Some(10),
            ..Default::default()
        });
        Ok(resp)
    }

    async fn get_streaming_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatStream> {
        Ok(Box::pin(futures::stream::empty()))
    }

    fn model(&self) -> Option<&str> {
        Some("stub-model")
    }
}

/// A `tracing` layer that records each new span's name and a chosen field.
#[derive(Clone)]
struct SpanCapture {
    records: Arc<Mutex<Vec<(String, bool)>>>,
}

impl<S: tracing::Subscriber> Layer<S> for SpanCapture {
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        _id: &tracing::span::Id,
        _ctx: Context<'_, S>,
    ) {
        // Record whether the operation-name field is present on the span.
        let mut has_operation = false;
        struct Probe<'a>(&'a mut bool);
        impl tracing::field::Visit for Probe<'_> {
            fn record_str(&mut self, field: &tracing::field::Field, _value: &str) {
                if field.name() == attr::OPERATION {
                    *self.0 = true;
                }
            }
            fn record_debug(
                &mut self,
                field: &tracing::field::Field,
                _value: &dyn std::fmt::Debug,
            ) {
                if field.name() == attr::OPERATION {
                    *self.0 = true;
                }
            }
        }
        attrs.record(&mut Probe(&mut has_operation));
        self.records
            .lock()
            .unwrap()
            .push((attrs.metadata().name().to_string(), has_operation));
    }
}

#[test]
fn observable_chat_client_emits_chat_span() {
    let records = Arc::new(Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::registry().with(SpanCapture {
        records: records.clone(),
    });

    tracing::subscriber::with_default(subscriber, || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let client = ObservableChatClient::new(StubClient::default(), "stub");
            let resp = client
                .get_response(
                    vec![Message::user("hello")],
                    ChatOptions::new().with_model("test-model"),
                )
                .await
                .unwrap();
            assert_eq!(resp.text(), "hi");
        });
    });

    let captured = records.lock().unwrap();
    let chat_span = captured.iter().find(|(name, _)| name == "chat");
    assert!(
        chat_span.is_some(),
        "expected a 'chat' span, captured: {:?}",
        *captured
    );
    assert!(
        chat_span.unwrap().1,
        "chat span is missing the gen_ai.operation.name attribute"
    );
}

// ---------------------------------------------------------------------
// Richer field-value capture, for asserting exact attribute values added
// beyond the original smoke test above.
// ---------------------------------------------------------------------

#[derive(Debug, Default, Clone)]
struct CapturedSpan {
    name: &'static str,
    fields: HashMap<String, String>,
    /// The span id of this span's parent — explicit, or the contextually
    /// current span at creation — if any.
    parent: Option<u64>,
}

/// A `tracing` layer that records every field (by stringified value) set on
/// every span, both at creation (`on_new_span`) and via later `.record()`
/// calls (`on_record`) — most of the attributes under test here are filled
/// in after span creation (`record_request`/`record_response`/`record_error`
/// etc. all `.record()` onto initially-`Empty` fields).
#[derive(Clone, Default)]
struct FieldCapture {
    spans: Arc<Mutex<HashMap<u64, CapturedSpan>>>,
}

impl FieldCapture {
    /// The most recently created span with the given (static metadata) name.
    fn by_name(&self, name: &str) -> Option<CapturedSpan> {
        self.spans
            .lock()
            .unwrap()
            .values()
            .find(|s| s.name == name)
            .cloned()
    }

    /// Every captured span with the given (static metadata) name.
    fn all_by_name(&self, name: &str) -> Vec<CapturedSpan> {
        self.spans
            .lock()
            .unwrap()
            .values()
            .filter(|s| s.name == name)
            .cloned()
            .collect()
    }

    /// The span id of the (single) captured span with the given name.
    fn id_by_name(&self, name: &str) -> Option<u64> {
        self.spans
            .lock()
            .unwrap()
            .iter()
            .find(|(_, s)| s.name == name)
            .map(|(id, _)| *id)
    }
}

struct Recorder<'a>(&'a mut HashMap<String, String>);

impl tracing::field::Visit for Recorder<'_> {
    fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
        self.0.insert(field.name().to_string(), value.to_string());
    }
    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        self.0.insert(field.name().to_string(), value.to_string());
    }
    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        self.0.insert(field.name().to_string(), value.to_string());
    }
    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        self.0.insert(field.name().to_string(), value.to_string());
    }
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.0.insert(field.name().to_string(), value.to_string());
    }
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0
            .insert(field.name().to_string(), format!("{value:?}"));
    }
}

impl<S: tracing::Subscriber> Layer<S> for FieldCapture {
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        id: &tracing::span::Id,
        ctx: Context<'_, S>,
    ) {
        let mut fields = HashMap::new();
        attrs.record(&mut Recorder(&mut fields));
        // Parent resolution mirrors tracing's own: an explicitly assigned
        // parent wins; a contextual (non-root) span parents onto whatever
        // span is current on this thread at creation time.
        let parent = attrs
            .parent()
            .cloned()
            .or_else(|| {
                if attrs.is_contextual() {
                    ctx.current_span().id().cloned()
                } else {
                    None
                }
            })
            .map(|p| p.into_u64());
        self.spans.lock().unwrap().insert(
            id.into_u64(),
            CapturedSpan {
                name: attrs.metadata().name(),
                fields,
                parent,
            },
        );
    }

    fn on_record(
        &self,
        id: &tracing::span::Id,
        values: &tracing::span::Record<'_>,
        _ctx: Context<'_, S>,
    ) {
        let mut spans = self.spans.lock().unwrap();
        if let Some(entry) = spans.get_mut(&id.into_u64()) {
            values.record(&mut Recorder(&mut entry.fields));
        }
    }
}

/// Guards span-capturing tests in this binary: `tracing`'s callsite interest
/// cache is process-global, so running two capturing tests concurrently
/// (`cargo test`'s default) risks one test's subscriber not seeing spans
/// whose callsite interest was already resolved by a race with another
/// thread. Serializing via this mutex — one subscriber active at a time —
/// sidesteps that entirely. Deliberately a plain (not `tokio::sync`) mutex:
/// every test below holds it only across a *synchronous* `block_on`, never
/// across an `.await`, so there's no `await`-while-holding-a-lock hazard.
static SPAN_TEST_MUTEX: Mutex<()> = Mutex::new(());

/// Run `f` under a fresh capturing subscriber on a fresh current-thread
/// runtime, serialized against other span-capturing tests, and return what
/// was captured.
fn run_captured<F, Fut>(f: F) -> FieldCapture
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    let _guard = SPAN_TEST_MUTEX.lock().unwrap();
    let capture = FieldCapture::default();
    let subscriber = tracing_subscriber::registry().with(capture.clone());
    tracing::subscriber::with_default(subscriber, || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(f());
    });
    capture
}

fn parsed_f64(span: &CapturedSpan, key: &str) -> f64 {
    span.fields
        .get(key)
        .unwrap_or_else(|| panic!("missing field {key}, present: {:?}", span.fields.keys()))
        .parse()
        .unwrap_or_else(|e| panic!("field {key} did not parse as f64: {e}"))
}

// -- chat span: request attributes -----------------------------------

#[test]
fn chat_span_records_request_attributes() {
    let capture = run_captured(|| async {
        let client = ObservableChatClient::new(StubClient::default(), "stub-provider");
        let mut options = ChatOptions::new()
            .with_model("test-model")
            .with_temperature(0.5);
        options.top_p = Some(0.25);
        options.max_tokens = Some(128);
        options.seed = Some(42);
        options.frequency_penalty = Some(0.75);
        options.presence_penalty = Some(0.125);
        options.stop = Some(vec!["STOP".to_string(), "END".to_string()]);
        options.conversation_id = Some("conv-123".to_string());
        let _ = client
            .get_response(vec![Message::user("hello")], options)
            .await;
    });

    let span = capture.by_name("chat").expect("expected a chat span");
    assert!((parsed_f64(&span, attr::REQUEST_TEMPERATURE) - 0.5).abs() < 1e-9);
    assert!((parsed_f64(&span, attr::REQUEST_TOP_P) - 0.25).abs() < 1e-9);
    assert_eq!(
        span.fields
            .get(attr::REQUEST_MAX_TOKENS)
            .map(String::as_str),
        Some("128")
    );
    assert_eq!(
        span.fields.get(attr::REQUEST_SEED).map(String::as_str),
        Some("42")
    );
    assert!((parsed_f64(&span, attr::REQUEST_FREQUENCY_PENALTY) - 0.75).abs() < 1e-9);
    assert!((parsed_f64(&span, attr::REQUEST_PRESENCE_PENALTY) - 0.125).abs() < 1e-9);
    let stop = span
        .fields
        .get(attr::REQUEST_STOP_SEQUENCES)
        .expect("stop sequences present");
    assert!(stop.contains("STOP") && stop.contains("END"), "got: {stop}");
    assert_eq!(
        span.fields.get(attr::CONVERSATION_ID).map(String::as_str),
        Some("conv-123")
    );
    // gen_ai.request.model is still set directly, unaffected by the new
    // request-attribute recording.
    assert_eq!(
        span.fields.get(attr::REQUEST_MODEL).map(String::as_str),
        Some("test-model")
    );
}

#[test]
fn chat_span_omits_unset_request_attributes() {
    let capture = run_captured(|| async {
        let client = ObservableChatClient::new(StubClient::default(), "stub-provider");
        let _ = client
            .get_response(
                vec![Message::user("hello")],
                ChatOptions::new().with_model("test-model"),
            )
            .await;
    });
    let span = capture.by_name("chat").expect("expected a chat span");
    for key in [
        attr::REQUEST_TEMPERATURE,
        attr::REQUEST_TOP_P,
        attr::REQUEST_MAX_TOKENS,
        attr::REQUEST_SEED,
        attr::REQUEST_FREQUENCY_PENALTY,
        attr::REQUEST_PRESENCE_PENALTY,
        attr::REQUEST_STOP_SEQUENCES,
        attr::CONVERSATION_ID,
    ] {
        assert!(
            !span.fields.contains_key(key),
            "expected {key} to be absent when unset, got {:?}",
            span.fields.get(key)
        );
    }
}

// -- chat span: gen_ai.system / gen_ai.provider.name dual emission ----

#[test]
fn chat_span_dual_emits_system_and_provider_name() {
    let capture = run_captured(|| async {
        let client = ObservableChatClient::new(StubClient::default(), "my-provider");
        let _ = client
            .get_response(
                vec![Message::user("hi")],
                ChatOptions::new().with_model("m"),
            )
            .await;
    });
    let span = capture.by_name("chat").expect("expected a chat span");
    assert_eq!(
        span.fields.get(attr::SYSTEM).map(String::as_str),
        Some("my-provider")
    );
    assert_eq!(
        span.fields.get(attr::PROVIDER_NAME).map(String::as_str),
        Some("my-provider")
    );
}

// -- chat span: response model ----------------------------------------

#[test]
fn chat_span_records_response_model() {
    let capture = run_captured(|| async {
        let client = ObservableChatClient::new(
            StubClient {
                fail: false,
                response_model: Some("resolved-model".to_string()),
            },
            "p",
        );
        let _ = client
            .get_response(
                vec![Message::user("hi")],
                ChatOptions::new().with_model("m"),
            )
            .await;
    });
    let span = capture.by_name("chat").expect("expected a chat span");
    assert_eq!(
        span.fields.get(attr::RESPONSE_MODEL).map(String::as_str),
        Some("resolved-model")
    );
}

// -- chat span: content-capture gating ---------------------------------

fn options_with_tool() -> ChatOptions {
    let tool = FunctionTool::new(
        "get_weather",
        "gets the weather",
        serde_json::json!({"type": "object", "properties": {}}),
        |_args| async move { Ok(serde_json::json!({"temp_f": 65})) },
    )
    .into_definition();
    ChatOptions::new()
        .with_model("m")
        .with_instructions("be nice")
        .with_tool(tool)
}

#[test]
fn chat_span_content_capture_disabled_omits_gated_attributes() {
    let capture = run_captured(|| async {
        let client = ObservableChatClient::new(StubClient::default(), "p");
        let _ = client
            .get_response(vec![Message::user("hi")], options_with_tool())
            .await;
    });
    let span = capture.by_name("chat").expect("expected a chat span");
    assert!(!span.fields.contains_key(attr::SYSTEM_INSTRUCTIONS));
    assert!(!span.fields.contains_key(attr::TOOL_DEFINITIONS));
    assert!(!span.fields.contains_key(attr::INPUT_MESSAGES));
    assert!(!span.fields.contains_key(attr::OUTPUT_MESSAGES));
}

#[test]
fn chat_span_content_capture_enabled_includes_gated_attributes() {
    let capture = run_captured(|| async {
        let client =
            ObservableChatClient::new(StubClient::default(), "p").with_content_capture(true);
        let _ = client
            .get_response(vec![Message::user("hi")], options_with_tool())
            .await;
    });
    let span = capture.by_name("chat").expect("expected a chat span");
    let instructions = span
        .fields
        .get(attr::SYSTEM_INSTRUCTIONS)
        .expect("system instructions present");
    assert!(instructions.contains("be nice"), "got: {instructions}");
    let tools = span
        .fields
        .get(attr::TOOL_DEFINITIONS)
        .expect("tool definitions present");
    assert!(tools.contains("get_weather"), "got: {tools}");
    assert!(span.fields.contains_key(attr::INPUT_MESSAGES));
    assert!(span.fields.contains_key(attr::OUTPUT_MESSAGES));
}

// -- chat span: error path ----------------------------------------------

#[test]
fn chat_span_records_error_status_on_failure() {
    let capture = run_captured(|| async {
        let client = ObservableChatClient::new(
            StubClient {
                fail: true,
                response_model: None,
            },
            "p",
        );
        let _ = client
            .get_response(
                vec![Message::user("hi")],
                ChatOptions::new().with_model("m"),
            )
            .await;
    });
    let span = capture.by_name("chat").expect("expected a chat span");
    assert_eq!(
        span.fields.get(attr::ERROR_TYPE).map(String::as_str),
        Some("service")
    );
    assert_eq!(
        span.fields.get(attr::OTEL_STATUS_CODE).map(String::as_str),
        Some("ERROR")
    );
    let message = span
        .fields
        .get(attr::OTEL_STATUS_MESSAGE)
        .expect("status message present");
    assert!(message.contains("boom"), "got: {message}");
}

// -- tool span --------------------------------------------------------

#[test]
fn tool_span_ex_records_description_type_and_gated_arguments_result() {
    let capture = run_captured(|| async {
        let span = tool_span_ex("get_weather", "call-1", Some("gets the weather"));
        async {
            let args = serde_json::json!({"city": "Seattle"});
            let result = serde_json::json!({"temp_f": 65});
            let current = tracing::Span::current();
            record_tool_arguments(&current, &args, true);
            record_tool_result(&current, &result, true);
        }
        .instrument(span)
        .await;
    });

    let span = capture
        .by_name("execute_tool")
        .expect("expected an execute_tool span");
    assert_eq!(
        span.fields.get(attr::TOOL_NAME).map(String::as_str),
        Some("get_weather")
    );
    assert_eq!(
        span.fields.get(attr::TOOL_CALL_ID).map(String::as_str),
        Some("call-1")
    );
    assert_eq!(
        span.fields.get(attr::TOOL_DESCRIPTION).map(String::as_str),
        Some("gets the weather")
    );
    assert_eq!(
        span.fields.get(attr::TOOL_TYPE).map(String::as_str),
        Some("function")
    );
    let args_captured = span
        .fields
        .get(attr::TOOL_CALL_ARGUMENTS)
        .expect("arguments captured");
    assert!(args_captured.contains("Seattle"), "got: {args_captured}");
    let result_captured = span
        .fields
        .get(attr::TOOL_CALL_RESULT)
        .expect("result captured");
    assert!(result_captured.contains("65"), "got: {result_captured}");
}

#[test]
fn tool_span_gates_arguments_and_result_behind_content_capture() {
    let capture = run_captured(|| async {
        let span = tool_span_ex("get_weather", "call-1", None);
        async {
            let args = serde_json::json!({"city": "Seattle"});
            let result = serde_json::json!({"temp_f": 65});
            let current = tracing::Span::current();
            record_tool_arguments(&current, &args, false);
            record_tool_result(&current, &result, false);
        }
        .instrument(span)
        .await;
    });
    let span = capture
        .by_name("execute_tool")
        .expect("expected an execute_tool span");
    assert!(!span.fields.contains_key(attr::TOOL_CALL_ARGUMENTS));
    assert!(!span.fields.contains_key(attr::TOOL_CALL_RESULT));
    assert!(!span.fields.contains_key(attr::TOOL_DESCRIPTION));
}

#[test]
fn tool_span_delegates_to_tool_span_ex_with_function_type() {
    let capture = run_captured(|| async {
        let span = tool_span("noop", "call-2");
        async {}.instrument(span).await;
    });
    let span = capture
        .by_name("execute_tool")
        .expect("expected an execute_tool span");
    assert_eq!(
        span.fields.get(attr::TOOL_TYPE).map(String::as_str),
        Some("function")
    );
    assert_eq!(
        span.fields.get(attr::TOOL_NAME).map(String::as_str),
        Some("noop")
    );
    assert!(!span.fields.contains_key(attr::TOOL_DESCRIPTION));
}

#[test]
fn tool_span_records_error_status() {
    let capture = run_captured(|| async {
        let span = tool_span_ex("flaky", "call-3", None);
        async {
            record_error(&tracing::Span::current(), &Error::tool("kaboom"));
        }
        .instrument(span)
        .await;
    });
    let span = capture
        .by_name("execute_tool")
        .expect("expected an execute_tool span");
    assert_eq!(
        span.fields.get(attr::ERROR_TYPE).map(String::as_str),
        Some("tool")
    );
    assert_eq!(
        span.fields.get(attr::OTEL_STATUS_CODE).map(String::as_str),
        Some("ERROR")
    );
    let message = span
        .fields
        .get(attr::OTEL_STATUS_MESSAGE)
        .expect("status message present");
    assert!(message.contains("kaboom"), "got: {message}");
}

// ---------------------------------------------------------------------
// Parallel tool calls: every execute_tool span keeps the surrounding span
// as its parent. Regression guard mirroring upstream's "preserve tool span
// context for parallel calls" fix (microsoft/agent-framework#6512): Python
// lost the ambient span when fanning tool calls out via asyncio.create_task
// without copying contextvars; the Rust loop polls all invocations in-task
// under the instrumented future, so the parent must always propagate.
// ---------------------------------------------------------------------

/// A scripted client whose first response requests TWO parallel tool calls
/// and whose second response is the final answer.
#[derive(Clone)]
struct ParallelCallsClient {
    responses: Arc<Mutex<Vec<ChatResponse>>>,
}

impl ParallelCallsClient {
    fn new() -> Self {
        use agent_framework_core::types::FunctionArguments;
        let calls = vec![
            Content::FunctionCall(FunctionCallContent::new(
                "call_a",
                "alpha",
                Some(FunctionArguments::Raw("{}".into())),
            )),
            Content::FunctionCall(FunctionCallContent::new(
                "call_b",
                "beta",
                Some(FunctionArguments::Raw("{}".into())),
            )),
        ];
        let ask = ChatResponse {
            messages: vec![Message::with_contents(Role::assistant(), calls)],
            finish_reason: Some(FinishReason::tool_calls()),
            ..Default::default()
        };
        Self {
            responses: Arc::new(Mutex::new(vec![ask, ChatResponse::from_text("done")])),
        }
    }
}

#[async_trait]
impl ChatClient for ParallelCallsClient {
    async fn get_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        Ok(self.responses.lock().unwrap().remove(0))
    }

    async fn get_streaming_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatStream> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

#[test]
fn parallel_tool_call_spans_keep_the_surrounding_span_as_parent() {
    let capture = run_captured(|| async {
        // A tool that yields before answering, so the two invocations
        // genuinely interleave rather than completing back-to-back.
        let yielding_tool = |name: &str| {
            FunctionTool::new(
                name,
                "yields then answers",
                serde_json::json!({ "type": "object", "properties": {} }),
                |_args| async move {
                    tokio::task::yield_now().await;
                    Ok(serde_json::Value::String("ok".into()))
                },
            )
            .into_definition()
        };
        let client = FunctionInvokingChatClient::new(ParallelCallsClient::new());
        let options = ChatOptions {
            tools: vec![yielding_tool("alpha"), yielding_tool("beta")],
            ..Default::default()
        };
        let outer = tracing::info_span!("outer_agent_span");
        async {
            let resp = client
                .get_response(vec![Message::user("go")], options)
                .await
                .unwrap();
            assert_eq!(resp.text(), "done");
        }
        .instrument(outer)
        .await;
    });

    let outer_id = capture
        .id_by_name("outer_agent_span")
        .expect("expected the outer span to be captured");
    let tool_spans = capture.all_by_name("execute_tool");
    assert_eq!(
        tool_spans.len(),
        2,
        "expected one execute_tool span per parallel call"
    );
    for span in &tool_spans {
        assert_eq!(
            span.parent,
            Some(outer_id),
            "an execute_tool span lost its surrounding span (parent: {:?})",
            span.parent
        );
    }
}
