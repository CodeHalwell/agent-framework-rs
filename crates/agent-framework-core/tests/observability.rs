//! Isolated span-capture smoke test for [`ObservableChatClient`].
//!
//! This is a dedicated test binary on purpose: `tracing` caches callsite
//! interest globally, so if the `chat` span callsite were first evaluated under
//! the default no-op subscriber (in another test), it would be cached as
//! disabled and never fire under a thread-local capturing subscriber. Keeping
//! this the only test that touches the `chat` callsite makes the capture
//! deterministic.

use std::sync::{Arc, Mutex};

use agent_framework_core::observability::{attr, ObservableChatClient};
use agent_framework_core::prelude::*;
use async_trait::async_trait;
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::Layer;

/// A chat client that returns a single canned response and echoes usage/finish.
#[derive(Clone)]
struct StubClient;

#[async_trait]
impl ChatClient for StubClient {
    async fn get_response(
        &self,
        _messages: Vec<ChatMessage>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        let mut resp = ChatResponse::from_text("hi");
        resp.finish_reason = Some(FinishReason::stop());
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
        _messages: Vec<ChatMessage>,
        _options: ChatOptions,
    ) -> Result<ChatStream> {
        Ok(Box::pin(futures::stream::empty()))
    }

    fn model_id(&self) -> Option<&str> {
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
            let client = ObservableChatClient::new(StubClient, "stub");
            let resp = client
                .get_response(
                    vec![ChatMessage::user("hello")],
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
