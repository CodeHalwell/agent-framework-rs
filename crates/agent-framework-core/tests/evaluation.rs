//! The evaluation module (`experimental-evals`).
//!
//! Ported from upstream `test_evaluation.py` / `test_local_eval.py`.
#![cfg(feature = "experimental-evals")]

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use agent_framework_core::agent::SupportsAgentRun;
use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::evaluation::*;
use agent_framework_core::prelude::*;
use agent_framework_core::types::{FunctionArguments, FunctionCallContent};
use async_trait::async_trait;
use serde_json::json;

/// Answers each call with the next scripted response.
#[derive(Clone)]
struct Scripted(Arc<Mutex<VecDeque<ChatResponse>>>);

impl Scripted {
    fn new(responses: Vec<ChatResponse>) -> Self {
        Self(Arc::new(Mutex::new(responses.into())))
    }
}

#[async_trait]
impl ChatClient for Scripted {
    async fn get_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> agent_framework_core::error::Result<ChatResponse> {
        Ok(self
            .0
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| ChatResponse::from_text("fallback")))
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> agent_framework_core::error::Result<ChatStream> {
        use futures::StreamExt;
        let response = self.get_response(messages, options).await?;
        let updates: Vec<agent_framework_core::error::Result<ChatResponseUpdate>> = response
            .messages
            .into_iter()
            .map(|m| {
                Ok(ChatResponseUpdate {
                    contents: m.contents,
                    role: Some(m.role),
                    ..Default::default()
                })
            })
            .collect();
        Ok(futures::stream::iter(updates).boxed())
    }
}

fn agent(name: &str, replies: &[&str]) -> Arc<dyn SupportsAgentRun> {
    Arc::new(
        Agent::builder(Scripted::new(
            replies
                .iter()
                .map(|r| ChatResponse::from_text(*r))
                .collect(),
        ))
        .name(name)
        .build(),
    )
}

fn call_response(name: &str, args: serde_json::Value, text: &str) -> AgentResponse {
    let call = Message::with_contents(
        Role::assistant(),
        vec![Content::FunctionCall(FunctionCallContent::new(
            "c1",
            name,
            Some(FunctionArguments::Raw(args.to_string())),
        ))],
    );
    AgentResponse {
        messages: vec![call, Message::assistant(text)],
        ..Default::default()
    }
}

#[tokio::test]
async fn evaluate_agent_runs_queries_through_local_checks() {
    let weather = agent("weather", &["It is sunny today.", "Rain all week."]);
    let results = evaluate_agent()
        .agent(weather)
        .queries(["Weather today?", "Weather this week?"])
        .evaluator(keyword_check(["sunny"], false))
        .evaluator(evaluator("short", |item: &EvalItem| {
            item.response().len() < 100
        }))
        .run()
        .await
        .unwrap();
    assert_eq!(
        results.len(),
        1,
        "consecutive checks share one local evaluator"
    );
    let local = &results[0];
    assert_eq!(local.provider, "Local");
    assert_eq!(local.run_id, "Eval: weather");
    assert_eq!((local.passed(), local.failed()), (1, 1));
    assert_eq!(local.items[0].input_text.as_deref(), Some("Weather today?"));
    assert_eq!(local.per_evaluator["keyword_check"]["failed"], 1);
    assert_eq!(local.per_evaluator["short"]["passed"], 2);
    let err = local.raise_for_status(None).unwrap_err();
    assert!(err.0.contains("1 passed, 1 failed"), "{err}");
    assert!(local.assert_no_failed_items(None).is_err());
    assert!(local
        .assert_score_at_least(0.5, Some("short"), None)
        .is_ok());
    assert!(local.assert_score_at_least(0.5, None, None).is_err());
}

#[tokio::test]
async fn existing_responses_and_tool_expectations() {
    let responses = vec![
        call_response(
            "get_weather",
            json!({"location": "NYC", "units": "c"}),
            "Sunny in NYC",
        ),
        call_response("get_weather", json!({"location": "LA"}), "Sunny in LA"),
    ];
    let results =
        evaluate_agent()
            .queries(["NYC weather?", "Boston weather?"])
            .responses(responses)
            .expected_tool_calls(vec![
                vec![
                    ExpectedToolCall::new("get_weather").with_arguments(json!({"location": "NYC"}))
                ],
                vec![ExpectedToolCall::new("get_weather")
                    .with_arguments(json!({"location": "Boston"}))],
            ])
            .evaluator(tool_calls_present())
            .evaluator(tool_call_args_match())
            .evaluator(tool_called_check(
                ["get_weather", "get_time"],
                ToolCallMode::Any,
            ))
            .run()
            .await
            .unwrap();
    let r = &results[0];
    assert_eq!(r.items[0].status, "pass");
    assert_eq!(r.items[1].status, "fail");
    let reason = r.items[1].scores[1].sample.as_ref().unwrap()["reason"]
        .as_str()
        .unwrap();
    assert!(reason.contains("args mismatch"), "{reason}");
    assert!(reason.contains("\"LA\""), "{reason}");
}

#[tokio::test]
async fn argument_and_shape_validation() {
    let err = evaluate_agent()
        .queries(["q"])
        .evaluator(keyword_check(["x"], false))
        .run()
        .await
        .unwrap_err();
    assert!(err.to_string().contains("Provide 'agent'"), "{err}");
    let err = evaluate_agent()
        .agent(agent("a", &[]))
        .queries(["q1", "q2"])
        .expected_output(["only one"])
        .evaluator(keyword_check(["x"], false))
        .run()
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("2 queries but 1 expected_output"),
        "{err}"
    );
    let err = evaluate_agent()
        .agent(agent("a", &[]))
        .queries(["q"])
        .run()
        .await
        .unwrap_err();
    assert!(err.to_string().contains("at least one evaluator"), "{err}");
}

/// A provider that records what it was asked to evaluate.
struct Recording(Mutex<Vec<(String, usize)>>);

#[async_trait]
impl Evaluator for Recording {
    fn name(&self) -> &str {
        "recording"
    }
    async fn evaluate(
        &self,
        items: &[EvalItem],
        eval_name: &str,
    ) -> agent_framework_core::error::Result<EvalResults> {
        self.0
            .lock()
            .unwrap()
            .push((eval_name.to_string(), items.len()));
        let mut r = EvalResults::new("recording");
        r.result_counts = Some(
            [("passed".to_string(), items.len() as i64)]
                .into_iter()
                .collect(),
        );
        Ok(r)
    }
}

#[tokio::test]
async fn evaluate_workflow_scores_each_agent_and_the_output() {
    let workflow = agent_framework_core::workflow::SequentialBuilder::new()
        .participants(vec![
            agent("researcher", &["found facts"]),
            agent("writer", &["final draft"]),
        ])
        .build()
        .unwrap();
    let run = workflow.run("write about otters").await.unwrap();
    let recording = Arc::new(Recording(Mutex::new(Vec::new())));
    let results = evaluate_workflow(&workflow)
        .workflow_run(&run)
        .eval_name("WF")
        .evaluator(recording.clone() as Arc<dyn Evaluator>)
        .evaluator(keyword_check(["draft"], false))
        .run()
        .await
        .unwrap();
    assert_eq!(results.len(), 2);
    let rec = &results[0];
    assert_eq!(rec.sub_results.len(), 2, "{:?}", rec.sub_results.keys());
    assert!(rec.all_passed());
    let calls = recording.0.lock().unwrap().clone();
    assert!(
        calls
            .iter()
            .any(|(n, k)| n == "WF \u{2014} overall (recording)" && *k == 1),
        "{calls:?}"
    );
    // The local check sees the workflow's final output as the response.
    let local = &results[1];
    assert_eq!(local.passed(), 1, "{local:?}");
}
