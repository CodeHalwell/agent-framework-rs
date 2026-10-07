//! End-to-end tests of the experimental harness (`experimental-harness`):
//! the agent loop, standing tool approvals, and the todo / mode providers
//! driven through a real `Agent` over a scripted chat client.
//!
//! Ported from upstream `test_harness_loop.py`,
//! `test_harness_tool_approval.py`, `test_harness_todo.py` and
//! `test_harness_mode.py`.
#![cfg(feature = "experimental-harness")]

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use agent_framework_core::harness::*;
use agent_framework_core::prelude::*;
use agent_framework_core::types::{FunctionArguments, UsageContent};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

// region: scripted client

#[derive(Default)]
struct MockState {
    scripted: VecDeque<ChatResponse>,
    received: Vec<Vec<Message>>,
    options: Vec<ChatOptions>,
    service_mode: bool,
    conversations: usize,
    fail_next: bool,
}

/// Returns scripted responses in order, then "response to: <last text>".
#[derive(Clone, Default)]
struct Mock(Arc<Mutex<MockState>>);

impl Mock {
    fn new() -> Self {
        Self::default()
    }
    fn service() -> Self {
        let mock = Self::default();
        mock.0.lock().unwrap().service_mode = true;
        mock
    }
    fn texts(texts: &[&str]) -> Self {
        let mock = Self::new();
        for t in texts {
            mock.push(ChatResponse::from_text(*t));
        }
        mock
    }
    fn push(&self, response: ChatResponse) {
        self.0.lock().unwrap().scripted.push_back(response);
    }
    fn calls(&self) -> usize {
        self.0.lock().unwrap().received.len()
    }
    /// Make the next model call fail.
    fn fail_next(&self) {
        self.0.lock().unwrap().fail_next = true;
    }
    /// The message texts of call `i`.
    fn received(&self, i: usize) -> Vec<String> {
        self.0.lock().unwrap().received[i]
            .iter()
            .map(Message::text)
            .collect()
    }
    fn received_messages(&self, i: usize) -> Vec<Message> {
        self.0.lock().unwrap().received[i].clone()
    }
    fn options(&self, i: usize) -> ChatOptions {
        self.0.lock().unwrap().options[i].clone()
    }
}

#[async_trait]
impl ChatClient for Mock {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        let mut state = self.0.lock().unwrap();
        let last = messages.last().map(Message::text).unwrap_or_default();
        state.received.push(messages);
        state.options.push(options);
        if std::mem::take(&mut state.fail_next) {
            return Err(agent_framework_core::error::Error::Service(
                "scripted failure".into(),
            ));
        }
        let mut response = state
            .scripted
            .pop_front()
            .unwrap_or_else(|| ChatResponse::from_text(format!("response to: {last}")));
        if state.service_mode {
            state.conversations += 1;
            response.conversation_id = Some(format!("conv-{}", state.conversations));
        }
        Ok(response)
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let response = self.get_response(messages, options).await?;
        let conversation_id = response.conversation_id.clone();
        let last = response.messages.len().saturating_sub(1);
        let usage = response.usage_details.clone();
        let updates: Vec<Result<ChatResponseUpdate>> = response
            .messages
            .into_iter()
            .enumerate()
            .map(|(i, m)| {
                let mut contents = m.contents;
                // Usage rides the final update, as real providers stream it.
                if let (true, Some(details)) = (i == last, usage.clone()) {
                    contents.push(Content::Usage(UsageContent { details }));
                }
                Ok(ChatResponseUpdate {
                    contents,
                    role: Some(m.role),
                    message_id: Some(format!("m{i}")),
                    conversation_id: conversation_id.clone(),
                    ..Default::default()
                })
            })
            .collect();
        Ok(futures::stream::iter(updates).boxed())
    }
}

fn agent(client: &Mock) -> Arc<dyn SupportsAgentRun> {
    Arc::new(Agent::builder(client.clone()).build())
}

fn always(_: &LoopContext) -> bool {
    true
}

async fn collect(stream: agent_framework_core::agent::AgentRunStream) -> Vec<AgentResponseUpdate> {
    stream.map(|u| u.unwrap()).collect().await
}

fn any_contains(texts: &[String], needle: &str) -> bool {
    texts.iter().any(|t| t.contains(needle))
}

// region: loop basics

#[test]
fn rejects_zero_max_iterations() {
    let client = Mock::new();
    assert!(LoopAgent::builder(agent(&client), always)
        .max_iterations(Some(0))
        .build()
        .is_err());
}

#[tokio::test]
async fn loop_stops_at_max_iterations() {
    let client = Mock::new();
    let looping = LoopAgent::builder(agent(&client), always)
        .max_iterations(Some(3))
        .build()
        .unwrap();
    looping
        .run(vec![Message::user("start")], None)
        .await
        .unwrap();
    assert_eq!(client.calls(), 3);
}

#[tokio::test]
async fn default_cap_bounds_an_always_true_condition() {
    let client = Mock::new();
    let looping = LoopAgent::builder(agent(&client), always).build().unwrap();
    looping
        .run(vec![Message::user("start")], None)
        .await
        .unwrap();
    assert_eq!(client.calls(), DEFAULT_MAX_ITERATIONS);
}

#[tokio::test]
async fn condition_controls_iterations_and_receives_context() {
    let client = Mock::new();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen2 = seen.clone();
    let looping = LoopAgent::builder(agent(&client), move |ctx: &LoopContext| {
        seen2.lock().unwrap().push((
            ctx.iteration,
            ctx.original_messages[0].text(),
            ctx.last_result.text(),
        ));
        ctx.iteration < 2
    })
    .build()
    .unwrap();
    looping
        .run(vec![Message::user("start")], None)
        .await
        .unwrap();
    assert_eq!(client.calls(), 2);
    let seen = seen.lock().unwrap();
    assert_eq!(seen.iter().map(|s| s.0).collect::<Vec<_>>(), [1, 2]);
    assert_eq!(seen[0].1, "start");
    assert_eq!(seen[0].2, "response to: start");
}

#[tokio::test]
async fn async_condition_is_awaited() {
    let client = Mock::new();
    let looping = LoopAgent::builder(
        agent(&client),
        async_loop_callback(|ctx: LoopContext| async move {
            tokio::task::yield_now().await;
            Ok(LoopDecision::from(ctx.iteration < 3))
        }),
    )
    .build()
    .unwrap();
    looping
        .run(vec![Message::user("start")], None)
        .await
        .unwrap();
    assert_eq!(client.calls(), 3);
}

#[tokio::test]
async fn condition_errors_propagate() {
    let client = Mock::new();
    let looping = LoopAgent::builder(
        agent(&client),
        async_loop_callback(|_ctx: LoopContext| async move {
            Err::<LoopDecision, _>(Error::Other("boom".into()))
        }),
    )
    .build()
    .unwrap();
    assert!(looping
        .run(vec![Message::user("start")], None)
        .await
        .is_err());
}

#[tokio::test]
async fn default_next_message_nudge_is_used() {
    let client = Mock::new();
    let looping = LoopAgent::builder(agent(&client), always)
        .max_iterations(Some(2))
        .build()
        .unwrap();
    looping
        .run(vec![Message::user("original task")], None)
        .await
        .unwrap();
    assert!(any_contains(&client.received(0), "original task"));
    let second = client.received(1);
    assert_eq!(second.last().unwrap(), DEFAULT_NEXT_MESSAGE);
    // The default progress entry (the first answer) precedes the nudge.
    assert_eq!(
        second[second.len() - 2],
        "Progress so far:\n- response to: original task"
    );
}

#[tokio::test]
async fn custom_next_message_callable() {
    let client = Mock::new();
    let looping = LoopAgent::builder(agent(&client), always)
        .max_iterations(Some(2))
        .next_message(|ctx: &LoopContext| {
            Some(vec![Message::user(format!(
                "iteration {} follow-up",
                ctx.iteration
            ))])
        })
        .build()
        .unwrap();
    looping
        .run(vec![Message::user("task")], None)
        .await
        .unwrap();
    assert_eq!(client.received(1).last().unwrap(), "iteration 1 follow-up");
}

#[tokio::test]
async fn next_message_returning_none_reuses_messages() {
    let client = Mock::new();
    let looping = LoopAgent::builder(agent(&client), always)
        .max_iterations(Some(2))
        .next_message(|_: &LoopContext| None::<Vec<Message>>)
        .build()
        .unwrap();
    looping
        .run(vec![Message::user("only message")], None)
        .await
        .unwrap();
    let second = client.received(1);
    assert_eq!(second.last().unwrap(), "only message");
    assert!(!any_contains(&second, "Progress so far"));
}

#[tokio::test]
async fn non_streaming_returns_aggregated_transcript_by_default() {
    let client = Mock::texts(&["first answer", "second answer"]);
    let looping = LoopAgent::builder(agent(&client), always)
        .max_iterations(Some(2))
        .inject_progress(false)
        .build()
        .unwrap();
    let response = looping
        .run(vec![Message::user("start")], None)
        .await
        .unwrap();
    let text = response.text();
    assert!(text.contains("first answer"));
    assert!(text.contains("second answer"));
    assert!(text.contains(DEFAULT_NEXT_MESSAGE));
    let roles: Vec<&str> = response.messages.iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles, ["assistant", "user", "assistant"]);
}

#[tokio::test]
async fn aggregated_response_sums_usage() {
    let client = Mock::new();
    for text in ["a", "b"] {
        client.push(ChatResponse {
            usage_details: Some(UsageDetails {
                input_token_count: Some(10),
                output_token_count: Some(1),
                ..Default::default()
            }),
            ..ChatResponse::from_text(text)
        });
    }
    let looping = LoopAgent::builder(agent(&client), always)
        .max_iterations(Some(2))
        .build()
        .unwrap();
    let response = looping.run(vec![Message::user("go")], None).await.unwrap();
    let usage = response.usage_details.unwrap();
    assert_eq!(usage.input_token_count, Some(20));
    assert_eq!(usage.output_token_count, Some(2));
}

#[tokio::test]
async fn return_final_only_returns_last_response() {
    let client = Mock::texts(&["first answer", "second answer"]);
    let looping = LoopAgent::builder(agent(&client), always)
        .max_iterations(Some(2))
        .return_final_only(true)
        .build()
        .unwrap();
    let response = looping
        .run(vec![Message::user("start")], None)
        .await
        .unwrap();
    assert_eq!(response.text(), "second answer");
}

// region: feedback and progress

#[tokio::test]
async fn record_feedback_captures_and_injects_progress() {
    let client = Mock::new();
    let captured = Arc::new(Mutex::new(Vec::new()));
    let captured2 = captured.clone();
    let looping = LoopAgent::builder(agent(&client), always)
        .max_iterations(Some(3))
        .record_feedback(move |ctx: &LoopContext| {
            captured2.lock().unwrap().push(ctx.progress.clone());
            Some(format!("step-{}-done", ctx.iteration))
        })
        .build()
        .unwrap();
    looping
        .run(vec![Message::user("task")], None)
        .await
        .unwrap();
    assert_eq!(
        *captured.lock().unwrap(),
        vec![
            vec![],
            vec!["step-1-done".to_string()],
            vec!["step-1-done".to_string(), "step-2-done".to_string()],
        ]
    );
    assert!(any_contains(&client.received(1), "step-1-done"));
    // With a session, only the latest entry is injected; the earlier one is
    // already in the history.
    let third = client.received(2);
    assert_eq!(third[third.len() - 2], "Progress so far:\n- step-2-done");
    assert!(any_contains(&third, "step-1-done"));
}

#[tokio::test]
async fn feedback_flows_from_condition_to_callables() {
    let client = Mock::new();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen2 = seen.clone();
    let looping = LoopAgent::builder(agent(&client), |ctx: &LoopContext| {
        (
            ctx.iteration < 2,
            Some(format!("feedback-{}", ctx.iteration)),
        )
    })
    .record_feedback(move |ctx: &LoopContext| {
        seen2.lock().unwrap().push(ctx.feedback.clone());
        ctx.feedback.as_ref().map(|f| format!("logged-{f}"))
    })
    .next_message(|ctx: &LoopContext| {
        Some(vec![Message::user(format!(
            "address: {}",
            ctx.feedback.as_deref().unwrap_or_default()
        ))])
    })
    .build()
    .unwrap();
    looping
        .run(vec![Message::user("task")], None)
        .await
        .unwrap();
    assert_eq!(
        *seen.lock().unwrap(),
        [
            Some("feedback-1".to_string()),
            Some("feedback-2".to_string())
        ]
    );
    assert_eq!(client.received(1).last().unwrap(), "address: feedback-1");
}

#[tokio::test]
async fn plain_bool_condition_yields_no_feedback() {
    let client = Mock::new();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen2 = seen.clone();
    let looping = LoopAgent::builder(agent(&client), |ctx: &LoopContext| ctx.iteration < 2)
        .next_message(move |ctx: &LoopContext| {
            seen2.lock().unwrap().push(ctx.feedback.clone());
            Some(vec![Message::user("continue")])
        })
        .build()
        .unwrap();
    looping
        .run(vec![Message::user("task")], None)
        .await
        .unwrap();
    assert_eq!(*seen.lock().unwrap(), [None]);
}

#[tokio::test]
async fn inject_progress_false_exposes_progress_without_injecting() {
    let client = Mock::new();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen2 = seen.clone();
    let looping = LoopAgent::builder(agent(&client), move |ctx: &LoopContext| {
        seen2.lock().unwrap().push(ctx.progress.clone());
        ctx.iteration < 2
    })
    .inject_progress(false)
    .build()
    .unwrap();
    looping
        .run(vec![Message::user("task")], None)
        .await
        .unwrap();
    assert_eq!(
        *seen.lock().unwrap(),
        vec![vec![], vec!["response to: task".to_string()]]
    );
    assert!(!any_contains(&client.received(1), "Progress so far"));
}

#[tokio::test]
async fn marker_condition_stops_loop_early() {
    let client = Mock::texts(&["working", "all DONE"]);
    let looping = LoopAgent::builder(agent(&client), |ctx: &LoopContext| {
        !ctx.last_result.text().contains("DONE")
    })
    .build()
    .unwrap();
    looping
        .run(vec![Message::user("task")], None)
        .await
        .unwrap();
    assert_eq!(client.calls(), 2);
}

#[tokio::test]
async fn additional_instructions_are_prepended_as_a_system_message() {
    let client = Mock::new();
    let looping = LoopAgent::builder(agent(&client), always)
        .max_iterations(Some(1))
        .additional_instructions("Be thorough.")
        .build()
        .unwrap();
    looping
        .run(vec![Message::user("task")], None)
        .await
        .unwrap();
    let first = client.received_messages(0);
    assert_eq!(first[0].role, Role::system());
    assert_eq!(first[0].text(), "Be thorough.");
    assert_eq!(first[1].text(), "task");
}

// region: fresh context

#[tokio::test]
async fn fresh_context_resets_to_original_task_plus_progress() {
    let client = Mock::new();
    let looping = LoopAgent::builder(agent(&client), always)
        .max_iterations(Some(2))
        .fresh_context(true)
        .record_feedback(|ctx: &LoopContext| Some(format!("note-{}", ctx.iteration)))
        .build()
        .unwrap();
    looping
        .run(vec![Message::user("original task")], None)
        .await
        .unwrap();
    // The second pass sees the task, the progress log and the nudge, but not
    // the first pass's conversation.
    assert_eq!(
        client.received(1),
        [
            "original task",
            "Progress so far:\n- note-1",
            DEFAULT_NEXT_MESSAGE
        ]
    );
}

#[tokio::test]
async fn fresh_context_with_session_keeps_pre_loop_history_and_resets_state() {
    let client = Mock::new();
    let inner = agent(&client);
    let mut session = inner.create_session();
    inner
        .run(vec![Message::user("earlier turn")], Some(&mut session))
        .await
        .unwrap();
    session.state.insert("counter", json!(0));

    let looping = LoopAgent::builder(inner.clone(), |ctx: &LoopContext| {
        // Working state written during a pass is discarded before the next.
        let seen = ctx.session.state.get("counter").unwrap();
        ctx.session
            .state
            .insert("counter", json!(seen.as_i64().unwrap() + 1));
        assert_eq!(seen, json!(0));
        ctx.iteration < 3
    })
    .fresh_context(true)
    .inject_progress(false)
    .build()
    .unwrap();
    looping
        .run(vec![Message::user("loop task")], Some(&mut session))
        .await
        .unwrap();
    assert_eq!(client.calls(), 4);
    for call in 2..4 {
        assert_eq!(
            client.received(call),
            [
                "earlier turn",
                "response to: earlier turn",
                "loop task",
                DEFAULT_NEXT_MESSAGE
            ],
            "pass {call} must see only the pre-loop history plus its own input"
        );
    }
    // The last pass's state survives the loop.
    assert_eq!(session.state.get("counter"), Some(json!(1)));

    // Afterwards the session's history holds the pre-loop turn plus the final
    // pass, as upstream leaves it.
    inner
        .run(vec![Message::user("after")], Some(&mut session))
        .await
        .unwrap();
    assert_eq!(
        client.received(4),
        [
            "earlier turn",
            "response to: earlier turn",
            "loop task",
            DEFAULT_NEXT_MESSAGE,
            &format!("response to: {DEFAULT_NEXT_MESSAGE}"),
            "after"
        ]
    );
}

#[tokio::test]
async fn non_fresh_loop_accumulates_history() {
    let client = Mock::new();
    let looping = LoopAgent::builder(agent(&client), always)
        .max_iterations(Some(2))
        .inject_progress(false)
        .build()
        .unwrap();
    looping
        .run(vec![Message::user("task")], None)
        .await
        .unwrap();
    assert_eq!(
        client.received(1),
        ["task", "response to: task", DEFAULT_NEXT_MESSAGE]
    );
}

// region: judge

fn verdict(answered: bool, reasoning: &str) -> ChatResponse {
    ChatResponse::from_text(json!({"answered": answered, "reasoning": reasoning}).to_string())
}

#[tokio::test]
async fn judge_stops_when_answered_on_first_pass() {
    let client = Mock::new();
    let judge = Mock::new();
    judge.push(verdict(true, "complete"));
    let looping = LoopAgent::with_judge(agent(&client), Judge::new(Arc::new(judge.clone())))
        .build()
        .unwrap();
    looping
        .run(vec![Message::user("task")], None)
        .await
        .unwrap();
    assert_eq!(client.calls(), 1);
    assert_eq!(judge.calls(), 1);
}

#[tokio::test]
async fn judge_continues_until_answered_and_feeds_back_reasoning() {
    let client = Mock::new();
    let judge = Mock::new();
    judge.push(verdict(false, "missing the summary"));
    judge.push(verdict(true, "complete"));
    let looping = LoopAgent::with_judge(agent(&client), Judge::new(Arc::new(judge.clone())))
        .inject_progress(false)
        .build()
        .unwrap();
    looping
        .run(vec![Message::user("task")], None)
        .await
        .unwrap();
    assert_eq!(client.calls(), 2);
    let nudge = client.received(1).last().unwrap().clone();
    assert!(nudge.contains("Evaluator feedback: missing the summary"));
}

#[tokio::test]
async fn judge_requests_structured_output_and_sees_the_exchange() {
    let client = Mock::new();
    let judge = Mock::new();
    judge.push(verdict(true, "ok"));
    let looping = LoopAgent::with_judge(agent(&client), Judge::new(Arc::new(judge.clone())))
        .build()
        .unwrap();
    looping
        .run(vec![Message::user("task")], None)
        .await
        .unwrap();
    assert_eq!(
        judge.options(0).response_format,
        Some(JudgeVerdict::response_format())
    );
    let seen = judge.received(0);
    assert_eq!(seen[0], {
        let mut s = DEFAULT_JUDGE_INSTRUCTIONS.to_string();
        s = s.replace(CRITERIA_PLACEHOLDER, "");
        s
    });
    assert_eq!(
        &seen[1..],
        [
            "Evaluate the agent's work. The user's original request follows:",
            "task",
            "The agent's latest response was:",
            "response to: task",
            "Has the original request been fully addressed?"
        ]
    );
    // The judge sees no tools.
    assert!(judge.options(0).tools.is_empty());
}

#[tokio::test]
async fn judge_uses_structured_value() {
    let client = Mock::new();
    let judge = Mock::new();
    judge.push(ChatResponse {
        value: Some(json!({"answered": false, "reasoning": "more"})),
        ..ChatResponse::from_text("ignored")
    });
    judge.push(ChatResponse {
        value: Some(json!({"answered": true, "reasoning": "done"})),
        ..ChatResponse::from_text("ignored")
    });
    let looping = LoopAgent::with_judge(agent(&client), Judge::new(Arc::new(judge.clone())))
        .build()
        .unwrap();
    looping
        .run(vec![Message::user("task")], None)
        .await
        .unwrap();
    assert_eq!(client.calls(), 2);
}

#[tokio::test]
async fn judge_text_fallback_markers() {
    // MORE wins over DONE, and no marker keeps looping.
    for (replies, expected_calls) in [
        (
            vec![
                "Not yet. VERDICT: MORE (not VERDICT: DONE)",
                "VERDICT: DONE",
            ],
            2,
        ),
        (vec!["I am unsure.", "verdict: done"], 2),
        (vec!["VERDICT: DONE"], 1),
    ] {
        let client = Mock::new();
        let judge = Mock::texts(&replies);
        let looping = LoopAgent::with_judge(agent(&client), Judge::new(Arc::new(judge.clone())))
            .build()
            .unwrap();
        looping
            .run(vec![Message::user("task")], None)
            .await
            .unwrap();
        assert_eq!(client.calls(), expected_calls, "{replies:?}");
    }
}

#[tokio::test]
async fn judge_respects_default_max_iterations() {
    let client = Mock::new();
    let judge = Mock::new();
    for _ in 0..10 {
        judge.push(verdict(false, "again"));
    }
    let looping = LoopAgent::with_judge(agent(&client), Judge::new(Arc::new(judge.clone())))
        .build()
        .unwrap();
    looping
        .run(vec![Message::user("task")], None)
        .await
        .unwrap();
    assert_eq!(client.calls(), DEFAULT_JUDGE_MAX_ITERATIONS);
    // The cap short-circuits before the judge on the last pass.
    assert_eq!(judge.calls(), DEFAULT_JUDGE_MAX_ITERATIONS - 1);
}

#[tokio::test]
async fn judge_custom_parser_owns_interpretation() {
    let client = Mock::new();
    let judge = Mock::texts(&["yes", "whatever"]);
    let looping = LoopAgent::with_judge(
        agent(&client),
        Judge::new(Arc::new(judge.clone()))
            .response_format(None)
            .verdict_parser(|response: &ChatResponse| {
                Ok(JudgeVerdict {
                    answered: response.text() == "yes",
                    reasoning: String::new(),
                })
            }),
    )
    .build()
    .unwrap();
    looping
        .run(vec![Message::user("task")], None)
        .await
        .unwrap();
    assert_eq!(client.calls(), 1);
    assert_eq!(judge.options(0).response_format, None);

    // Parser errors surface instead of falling back to markers.
    let client = Mock::new();
    let judge = Mock::texts(&["VERDICT: DONE"]);
    let looping = LoopAgent::with_judge(
        agent(&client),
        Judge::new(Arc::new(judge))
            .verdict_parser(|_: &ChatResponse| Err(Error::Other("unparseable".into()))),
    )
    .build()
    .unwrap();
    assert!(looping
        .run(vec![Message::user("task")], None)
        .await
        .is_err());
}

#[tokio::test]
async fn judge_criteria_reach_agent_and_judge() {
    let client = Mock::new();
    let judge = Mock::new();
    judge.push(verdict(true, "ok"));
    let looping = LoopAgent::with_judge(
        agent(&client),
        Judge::new(Arc::new(judge.clone())).criteria(["cite sources"]),
    )
    .build()
    .unwrap();
    looping
        .run(vec![Message::user("task")], None)
        .await
        .unwrap();
    let first = client.received_messages(0);
    assert_eq!(first[0].role, Role::system());
    assert_eq!(
        first[0].text(),
        "Your response must satisfy all of the following criteria:\n- cite sources"
    );
    assert!(judge.received(0)[0]
        .ends_with("The response must satisfy all of the following criteria:\n- cite sources"));
}

// region: streaming

#[tokio::test]
async fn streaming_yields_each_pass_and_injected_messages() {
    let client = Mock::texts(&["first answer", "second answer"]);
    let looping = LoopAgent::builder(agent(&client), always)
        .max_iterations(Some(2))
        .build()
        .unwrap();
    let updates = collect(
        looping
            .run_stream(vec![Message::user("start")], None, None)
            .await
            .unwrap(),
    )
    .await;
    let summary: Vec<(String, String)> = updates
        .iter()
        .map(|u| {
            (
                u.role.as_ref().unwrap().as_str().to_string(),
                u.contents
                    .iter()
                    .filter_map(|c| match c {
                        Content::Text(t) => Some(t.text.clone()),
                        _ => None,
                    })
                    .collect::<String>(),
            )
        })
        .collect();
    assert_eq!(
        summary,
        [
            ("assistant".to_string(), "first answer".to_string()),
            (
                "user".to_string(),
                "Progress so far:\n- first answer".to_string()
            ),
            ("user".to_string(), DEFAULT_NEXT_MESSAGE.to_string()),
            ("assistant".to_string(), "second answer".to_string()),
        ]
    );
    // Injected messages carry distinct message ids so they aggregate apart.
    assert_ne!(updates[1].message_id, updates[2].message_id);
    assert_eq!(client.calls(), 2);
}

#[tokio::test]
async fn streaming_carries_service_conversation_id_between_passes() {
    let client = Mock::service();
    let looping = LoopAgent::builder(agent(&client), always)
        .max_iterations(Some(3))
        .build()
        .unwrap();
    let session = AgentSession::new();
    collect(
        looping
            .run_stream(vec![Message::user("start")], Some(session), None)
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(client.options(0).conversation_id, None);
    assert_eq!(client.options(1).conversation_id.as_deref(), Some("conv-1"));
    assert_eq!(client.options(2).conversation_id.as_deref(), Some("conv-2"));
}

// region: approval escape hatch

fn call_response(calls: &[(&str, &str, &str)]) -> ChatResponse {
    ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            calls
                .iter()
                .map(|(id, name, args)| {
                    Content::FunctionCall(FunctionCallContent::new(
                        *id,
                        *name,
                        Some(FunctionArguments::Raw((*args).into())),
                    ))
                })
                .collect(),
        )],
        finish_reason: Some(FinishReason::tool_calls()),
        ..Default::default()
    }
}

fn counting_tool(name: &str, counter: Arc<Mutex<Vec<Value>>>) -> ToolDefinition {
    FunctionTool::new(
        name,
        format!("The {name} tool."),
        json!({"type": "object", "properties": {"value": {"type": "string"}}}),
        move |args| {
            let counter = counter.clone();
            async move {
                counter.lock().unwrap().push(args);
                Ok(json!("ok"))
            }
        },
    )
    .with_approval_mode(ApprovalMode::AlwaysRequire)
    .into_definition()
}

fn approval_requests(response: &AgentResponse) -> Vec<FunctionApprovalRequestContent> {
    response
        .messages
        .iter()
        .flat_map(|m| &m.contents)
        .filter_map(|c| match c {
            Content::FunctionApprovalRequest(r) => Some(r.clone()),
            _ => None,
        })
        .collect()
}

fn approve(request: &FunctionApprovalRequestContent) -> Message {
    Message::with_contents(
        Role::user(),
        vec![Content::FunctionApprovalResponse(
            request.create_response(true),
        )],
    )
}

#[tokio::test]
async fn loop_stops_on_pending_approval_request() {
    let client = Mock::new();
    client.push(call_response(&[("c1", "guarded", "{}")]));
    let calls = Arc::new(Mutex::new(Vec::new()));
    let inner: Arc<dyn SupportsAgentRun> = Arc::new(
        Agent::builder(client.clone())
            .tool(counting_tool("guarded", calls))
            .build(),
    );
    let looping = LoopAgent::builder(inner, always).build().unwrap();
    let response = looping.run(vec![Message::user("go")], None).await.unwrap();
    assert_eq!(client.calls(), 1);
    assert_eq!(approval_requests(&response).len(), 1);

    let client = Mock::new();
    client.push(call_response(&[("c1", "guarded", "{}")]));
    let inner: Arc<dyn SupportsAgentRun> = Arc::new(
        Agent::builder(client.clone())
            .tool(counting_tool("guarded", Arc::new(Mutex::new(Vec::new()))))
            .build(),
    );
    let looping = LoopAgent::builder(inner, always).build().unwrap();
    let updates = collect(
        looping
            .run_stream(vec![Message::user("go")], None, None)
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(client.calls(), 1);
    assert!(updates
        .iter()
        .flat_map(|u| &u.contents)
        .any(|c| matches!(c, Content::FunctionApprovalRequest(_))));
}

// region: tool approval

struct ApprovalFixture {
    client: Mock,
    calls: Arc<Mutex<Vec<(String, Value)>>>,
    agent: ToolApprovalAgent,
    session: AgentSession,
}

fn approval_fixture(
    tools: &[&str],
    configure: impl FnOnce(ToolApprovalAgent) -> ToolApprovalAgent,
) -> ApprovalFixture {
    let client = Mock::new();
    let calls: Arc<Mutex<Vec<(String, Value)>>> = Arc::new(Mutex::new(Vec::new()));
    let mut builder = Agent::builder(client.clone());
    for name in tools {
        let calls = calls.clone();
        let tool_name = name.to_string();
        builder = builder.tool(
            FunctionTool::new(
                *name,
                format!("The {name} tool."),
                json!({"type": "object", "properties": {"value": {"type": "string"}}}),
                move |args| {
                    let calls = calls.clone();
                    let tool_name = tool_name.clone();
                    async move {
                        calls.lock().unwrap().push((tool_name, args));
                        Ok(json!("ok"))
                    }
                },
            )
            .with_approval_mode(ApprovalMode::AlwaysRequire)
            .into_definition(),
        );
    }
    let inner = Arc::new(builder.build());
    let session = inner.create_session();
    ApprovalFixture {
        client,
        calls,
        agent: configure(ToolApprovalAgent::new(inner)),
        session,
    }
}

impl ApprovalFixture {
    fn executed(&self) -> Vec<String> {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .map(|(n, _)| n.clone())
            .collect()
    }
    async fn run(&mut self, message: Message) -> AgentResponse {
        self.agent
            .run(vec![message], Some(&mut self.session))
            .await
            .unwrap()
    }
}

#[tokio::test]
async fn tool_approval_requires_a_session() {
    let f = approval_fixture(&[], |a| a);
    assert!(f.agent.run(vec![Message::user("hi")], None).await.is_err());
    assert!(f
        .agent
        .run_stream(vec![Message::user("hi")], None, None)
        .await
        .is_err());
}

#[tokio::test]
async fn tool_approval_presents_multiple_requests_one_at_a_time() {
    let mut f = approval_fixture(&["first_tool", "second_tool"], |a| a);
    f.client.push(call_response(&[
        ("call_first", "first_tool", "{}"),
        ("call_second", "second_tool", "{}"),
    ]));

    let first = f.run(Message::user("call both")).await;
    let requests = approval_requests(&first);
    assert_eq!(
        requests
            .iter()
            .map(|r| r.function_call.name.as_str())
            .collect::<Vec<_>>(),
        ["first_tool"]
    );

    let second = f.run(approve(&requests[0])).await;
    let requests = approval_requests(&second);
    assert_eq!(
        requests
            .iter()
            .map(|r| r.function_call.name.as_str())
            .collect::<Vec<_>>(),
        ["second_tool"]
    );
    // The queued request was answered from state: no model call, nothing ran.
    assert_eq!(f.client.calls(), 1);
    assert!(f.executed().is_empty());

    f.client.push(ChatResponse::from_text("done"));
    let last = f.run(approve(&requests[0])).await;
    assert_eq!(last.text(), "done");
    assert_eq!(f.executed(), ["first_tool", "second_tool"]);
}

#[tokio::test]
async fn auto_approval_rule_receives_function_call_and_approves_matches() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen2 = seen.clone();
    let mut f = approval_fixture(&["auto_write", "manual_write"], move |a| {
        a.with_auto_approval_rule(move |call: &FunctionCallContent| {
            seen2.lock().unwrap().push(call.name.clone());
            call.name == "auto_write"
        })
    });
    f.client.push(call_response(&[
        ("call_auto", "auto_write", "{}"),
        ("call_manual", "manual_write", "{}"),
    ]));
    let first = f.run(Message::user("write both")).await;
    let requests = approval_requests(&first);
    assert_eq!(
        requests
            .iter()
            .map(|r| r.function_call.name.as_str())
            .collect::<Vec<_>>(),
        ["manual_write"]
    );
    assert_eq!(*seen.lock().unwrap(), ["auto_write", "manual_write"]);
    assert!(f.executed().is_empty());

    f.client.push(ChatResponse::from_text("done"));
    let last = f.run(approve(&requests[0])).await;
    assert_eq!(last.text(), "done");
    let mut executed = f.executed();
    executed.sort();
    assert_eq!(executed, ["auto_write", "manual_write"]);
}

#[tokio::test]
async fn fully_auto_approved_batch_reruns_without_asking() {
    let mut f = approval_fixture(&["safe"], |a| {
        a.with_auto_approval_rule(|call: &FunctionCallContent| call.name == "safe")
    });
    f.client.push(call_response(&[("c1", "safe", "{}")]));
    f.client.push(ChatResponse::from_text("finished"));
    let response = f.run(Message::user("go")).await;
    assert_eq!(response.text(), "finished");
    assert!(approval_requests(&response).is_empty());
    assert_eq!(f.executed(), ["safe"]);
}

#[test]
fn auto_approval_cap_defaults_to_forty_and_rejects_zero() {
    let f = approval_fixture(&[], |a| a);
    assert_eq!(DEFAULT_MAX_AUTO_APPROVAL_ITERATIONS, 40);
    assert_eq!(f.agent.max_auto_approval_iterations(), 40);
    assert!(f
        .agent
        .clone()
        .with_max_auto_approval_iterations(0)
        .is_err());
}

/// A model that keeps calling an auto-approved tool.
fn endless_safe_calls(f: &ApprovalFixture, n: usize) {
    for i in 0..n {
        let id = format!("c{i}");
        f.client.push(call_response(&[(id.as_str(), "safe", "{}")]));
    }
}

#[tokio::test]
async fn auto_approval_reruns_stop_at_the_cap() {
    let mut f = approval_fixture(&["safe"], |a| {
        a.with_auto_approval_rule(|call: &FunctionCallContent| call.name == "safe")
            .with_max_auto_approval_iterations(2)
            .unwrap()
    });
    endless_safe_calls(&f, 10);
    let response = f.run(Message::user("go")).await;
    // Two auto-approved re-runs, then one final turn returned as-is.
    assert_eq!(f.client.calls(), 3);
    assert_eq!(f.executed().len(), 2);
    let requests = approval_requests(&response);
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].function_call.call_id, "c2");
}

#[tokio::test]
async fn streaming_auto_approval_reruns_stop_at_the_cap() {
    let f = approval_fixture(&["safe"], |a| {
        a.with_auto_approval_rule(|call: &FunctionCallContent| call.name == "safe")
            .with_max_auto_approval_iterations(2)
            .unwrap()
    });
    endless_safe_calls(&f, 10);
    let updates = collect(
        f.agent
            .run_stream(vec![Message::user("go")], Some(f.session.clone()), None)
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(f.client.calls(), 3);
    assert_eq!(f.executed().len(), 2);
    let requests: Vec<_> = updates
        .iter()
        .flat_map(|u| &u.contents)
        .filter_map(|c| match c {
            Content::FunctionApprovalRequest(r) => Some(r.function_call.call_id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(requests, ["c2"]);
}

fn with_usage(response: ChatResponse, input: u64, output: u64) -> ChatResponse {
    ChatResponse {
        usage_details: Some(UsageDetails {
            input_token_count: Some(input),
            output_token_count: Some(output),
            ..Default::default()
        }),
        ..response
    }
}

/// Two auto-approved passes (10/1 and 20/2 tokens), then a final turn
/// (40/4) that is returned or capped.
fn auto_approval_usage_script(f: &ApprovalFixture) {
    f.client
        .push(with_usage(call_response(&[("c0", "safe", "{}")]), 10, 1));
    f.client
        .push(with_usage(call_response(&[("c1", "safe", "{}")]), 20, 2));
    f.client
        .push(with_usage(ChatResponse::from_text("finished"), 40, 4));
}

#[tokio::test]
async fn auto_approval_reruns_sum_usage() {
    let mut f = approval_fixture(&["safe"], |a| {
        a.with_auto_approval_rule(|call: &FunctionCallContent| call.name == "safe")
    });
    auto_approval_usage_script(&f);
    let response = f.run(Message::user("go")).await;
    assert_eq!(response.text(), "finished");
    assert_eq!(f.client.calls(), 3);
    let usage = response.usage_details.unwrap();
    assert_eq!(usage.input_token_count, Some(70));
    assert_eq!(usage.output_token_count, Some(7));

    // Hitting the cap returns the last pass as-is, still with the full sum.
    let mut f = approval_fixture(&["safe"], |a| {
        a.with_auto_approval_rule(|call: &FunctionCallContent| call.name == "safe")
            .with_max_auto_approval_iterations(1)
            .unwrap()
    });
    auto_approval_usage_script(&f);
    let response = f.run(Message::user("go")).await;
    assert_eq!(f.client.calls(), 2);
    assert_eq!(approval_requests(&response).len(), 1);
    let usage = response.usage_details.unwrap();
    assert_eq!(usage.input_token_count, Some(30));
    assert_eq!(usage.output_token_count, Some(3));
}

#[tokio::test]
async fn streaming_auto_approval_reruns_sum_usage() {
    let f = approval_fixture(&["safe"], |a| {
        a.with_auto_approval_rule(|call: &FunctionCallContent| call.name == "safe")
    });
    auto_approval_usage_script(&f);
    let updates = collect(
        f.agent
            .run_stream(vec![Message::user("go")], Some(f.session.clone()), None)
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(f.client.calls(), 3);
    let response = AgentResponse::from_updates(updates);
    assert_eq!(response.text(), "finished");
    assert!(approval_requests(&response).is_empty());
    let usage = response.usage_details.unwrap();
    assert_eq!(usage.input_token_count, Some(70));
    assert_eq!(usage.output_token_count, Some(7));
}

#[tokio::test]
async fn collected_approvals_survive_a_failed_inner_run() {
    let mut f = approval_fixture(&["first_tool", "second_tool"], |a| a);
    f.client.push(call_response(&[
        ("call_first", "first_tool", "{}"),
        ("call_second", "second_tool", "{}"),
    ]));
    let first = f.run(Message::user("call both")).await;
    let second = f.run(approve(&approval_requests(&first)[0])).await;
    let last_request = approval_requests(&second).remove(0);

    f.client.fail_next();
    assert!(f
        .agent
        .run(vec![approve(&last_request)], Some(&mut f.session))
        .await
        .is_err());
    assert_eq!(
        f.agent
            .state(&f.session)
            .unwrap()
            .collected_approval_responses
            .len(),
        2
    );

    // The batch ran before the model call failed; nothing of that run was
    // stored, so retrying the last answer sends (and runs) the whole batch.
    assert_eq!(f.executed(), ["first_tool", "second_tool"]);
    f.client.push(ChatResponse::from_text("done"));
    let last = f.run(approve(&last_request)).await;
    assert_eq!(last.text(), "done");
    assert_eq!(
        f.executed(),
        ["first_tool", "second_tool", "first_tool", "second_tool"]
    );
    assert!(f
        .agent
        .state(&f.session)
        .unwrap()
        .collected_approval_responses
        .is_empty());
}

#[tokio::test]
async fn streaming_collected_approvals_survive_a_failed_inner_run() {
    let mut f = approval_fixture(&["first_tool", "second_tool"], |a| a);
    f.client.push(call_response(&[
        ("call_first", "first_tool", "{}"),
        ("call_second", "second_tool", "{}"),
    ]));
    let first = f.run(Message::user("call both")).await;
    let second = f.run(approve(&approval_requests(&first)[0])).await;
    let last_request = approval_requests(&second).remove(0);

    f.client.fail_next();
    let results: Vec<_> = f
        .agent
        .run_stream(vec![approve(&last_request)], Some(f.session.clone()), None)
        .await
        .unwrap()
        .collect()
        .await;
    assert!(results.iter().any(|r| r.is_err()));
    assert_eq!(f.executed(), ["first_tool", "second_tool"]);
    assert_eq!(
        f.agent
            .state(&f.session)
            .unwrap()
            .collected_approval_responses
            .len(),
        2
    );

    f.client.push(ChatResponse::from_text("done"));
    let updates = collect(
        f.agent
            .run_stream(vec![approve(&last_request)], Some(f.session.clone()), None)
            .await
            .unwrap(),
    )
    .await;
    assert!(updates
        .iter()
        .flat_map(|u| &u.contents)
        .any(|c| matches!(c, Content::Text(t) if t.text == "done")));
    assert_eq!(
        f.executed(),
        ["first_tool", "second_tool", "first_tool", "second_tool"]
    );
}

#[tokio::test]
async fn always_approve_tool_adds_a_standing_rule() {
    let mut f = approval_fixture(&["dangerous_tool"], |a| a);
    f.client.push(call_response(&[(
        "call_initial",
        "dangerous_tool",
        r#"{"value": "one"}"#,
    )]));
    let first = f.run(Message::user("call once")).await;
    let request = approval_requests(&first).remove(0);

    f.client.push(ChatResponse::from_text("first done"));
    let response = f
        .agent
        .always_approve_tool_response(&f.session, &request, Some("trusted"))
        .unwrap();
    assert!(response.approved);
    f.run(Message::with_contents(
        Role::user(),
        vec![Content::FunctionApprovalResponse(response)],
    ))
    .await;
    assert_eq!(f.executed().len(), 1);
    assert_eq!(
        f.agent.state(&f.session).unwrap().rules,
        [ToolApprovalRule::tool("dangerous_tool").unwrap()]
    );

    f.client.push(call_response(&[(
        "call_auto",
        "dangerous_tool",
        r#"{"value": "two"}"#,
    )]));
    f.client.push(ChatResponse::from_text("second done"));
    let second = f.run(Message::user("call again")).await;
    assert!(approval_requests(&second).is_empty());
    assert_eq!(second.text(), "second done");
    assert_eq!(f.executed().len(), 2);
}

#[tokio::test]
async fn always_approve_with_arguments_only_covers_identical_arguments() {
    let mut f = approval_fixture(&["args_tool"], |a| a);
    f.client.push(call_response(&[(
        "c1",
        "args_tool",
        r#"{"value": "same"}"#,
    )]));
    let first = f.run(Message::user("first")).await;
    let request = approval_requests(&first).remove(0);
    let response = f
        .agent
        .always_approve_tool_with_arguments_response(&f.session, &request, None)
        .unwrap();
    f.client.push(ChatResponse::from_text("ok"));
    f.run(Message::with_contents(
        Role::user(),
        vec![Content::FunctionApprovalResponse(response)],
    ))
    .await;
    assert_eq!(f.executed().len(), 1);

    // Same arguments: approved by the rule.
    f.client
        .push(call_response(&[("c2", "args_tool", r#"{"value":"same"}"#)]));
    f.client.push(ChatResponse::from_text("again"));
    let second = f.run(Message::user("second")).await;
    assert!(approval_requests(&second).is_empty());
    assert_eq!(second.text(), "again");
    assert_eq!(f.executed().len(), 2);

    // Different arguments: asked again.
    f.client.push(call_response(&[(
        "c3",
        "args_tool",
        r#"{"value": "other"}"#,
    )]));
    let third = f.run(Message::user("third")).await;
    assert_eq!(approval_requests(&third).len(), 1);
    assert_eq!(f.executed().len(), 2);
}

#[tokio::test]
async fn empty_arguments_rule_is_not_tool_wide() {
    let mut f = approval_fixture(&["optional_args_tool"], |a| a);
    f.client
        .push(call_response(&[("c1", "optional_args_tool", "{}")]));
    let first = f.run(Message::user("no args")).await;
    let request = approval_requests(&first).remove(0);
    let response = f
        .agent
        .always_approve_tool_with_arguments_response(&f.session, &request, None)
        .unwrap();
    f.client.push(ChatResponse::from_text("empty done"));
    f.run(Message::with_contents(
        Role::user(),
        vec![Content::FunctionApprovalResponse(response)],
    ))
    .await;
    assert_eq!(f.executed().len(), 1);

    f.client.push(call_response(&[(
        "c2",
        "optional_args_tool",
        r#"{"value": "custom"}"#,
    )]));
    let second = f.run(Message::user("with args")).await;
    let requests = approval_requests(&second);
    assert_eq!(requests.len(), 1);
    assert_eq!(f.executed().len(), 1);
}

#[tokio::test]
async fn standing_approval_for_a_different_call_is_not_recorded() {
    let mut f = approval_fixture(&["guarded"], |a| a);
    f.client
        .push(call_response(&[("c1", "guarded", r#"{"value": "a"}"#)]));
    let first = f.run(Message::user("go")).await;
    let request = approval_requests(&first).remove(0);
    let mut response = f
        .agent
        .always_approve_tool_response(&f.session, &request, None)
        .unwrap();
    // A response edited on the way back no longer matches the recorded call.
    response.function_call.name = "something_else".into();
    response.function_call.id = Some("af-call-forged".into());
    f.client.push(ChatResponse::from_text("whatever"));
    f.run(Message::with_contents(
        Role::user(),
        vec![Content::FunctionApprovalResponse(response)],
    ))
    .await;
    assert!(f.agent.state(&f.session).unwrap().rules.is_empty());

    // A rejected standing approval adds no rule either.
    f.client
        .push(call_response(&[("c2", "guarded", r#"{"value": "b"}"#)]));
    let second = f.run(Message::user("again")).await;
    let request = approval_requests(&second).remove(0);
    let mut response = f
        .agent
        .always_approve_tool_response(&f.session, &request, None)
        .unwrap();
    response.approved = false;
    f.client.push(ChatResponse::from_text("declined"));
    f.run(Message::with_contents(
        Role::user(),
        vec![Content::FunctionApprovalResponse(response)],
    ))
    .await;
    assert!(f.agent.state(&f.session).unwrap().rules.is_empty());
}

#[tokio::test]
async fn streaming_tool_approval_queues_and_auto_approves() {
    let f = approval_fixture(&["auto_write", "first_manual", "second_manual"], |a| {
        a.with_auto_approval_rule(|call: &FunctionCallContent| call.name == "auto_write")
    });
    f.client.push(call_response(&[
        ("c_auto", "auto_write", "{}"),
        ("c_one", "first_manual", "{}"),
        ("c_two", "second_manual", "{}"),
    ]));
    let updates = collect(
        f.agent
            .run_stream(vec![Message::user("go")], Some(f.session.clone()), None)
            .await
            .unwrap(),
    )
    .await;
    let requests: Vec<FunctionApprovalRequestContent> = updates
        .iter()
        .flat_map(|u| &u.contents)
        .filter_map(|c| match c {
            Content::FunctionApprovalRequest(r) => Some(r.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        requests
            .iter()
            .map(|r| r.function_call.name.as_str())
            .collect::<Vec<_>>(),
        ["first_manual"]
    );
    let state = f.agent.state(&f.session).unwrap();
    assert_eq!(state.queued_approval_requests.len(), 1);
    assert_eq!(state.collected_approval_responses.len(), 1);

    // The queued request comes back next, without a model call.
    let updates = collect(
        f.agent
            .run_stream(vec![approve(&requests[0])], Some(f.session.clone()), None)
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(f.client.calls(), 1);
    let second: Vec<&FunctionApprovalRequestContent> = updates
        .iter()
        .flat_map(|u| &u.contents)
        .filter_map(|c| match c {
            Content::FunctionApprovalRequest(r) => Some(r),
            _ => None,
        })
        .collect();
    assert_eq!(second.len(), 1);
    assert_eq!(second[0].function_call.name, "second_manual");
    let last_request = second[0].clone();

    f.client.push(ChatResponse::from_text("all done"));
    let updates = collect(
        f.agent
            .run_stream(vec![approve(&last_request)], Some(f.session.clone()), None)
            .await
            .unwrap(),
    )
    .await;
    let text: String = updates
        .iter()
        .flat_map(|u| &u.contents)
        .filter_map(|c| match c {
            Content::Text(t) => Some(t.text.clone()),
            _ => None,
        })
        .collect();
    assert!(text.contains("all done"));
    let mut executed = f.executed();
    executed.sort();
    assert_eq!(executed, ["auto_write", "first_manual", "second_manual"]);
}

// region: providers through an agent

#[tokio::test]
async fn todo_provider_tools_run_through_an_agent() {
    let client = Mock::new();
    client.push(call_response(&[(
        "c1",
        "todos_add",
        r#"{"todos": [{"title": "Draft"}, {"title": "Review", "description": "twice"}]}"#,
    )]));
    client.push(ChatResponse::from_text("planned"));
    let todos = TodoProvider::new();
    let agent = Agent::builder(client.clone())
        .context_provider(Arc::new(todos.clone()))
        .build();
    let mut session = agent.create_session();
    let response = agent
        .run(vec![Message::user("plan it")], Some(&mut session))
        .await
        .unwrap();
    assert_eq!(response.text(), "planned");
    let items = todos.items(&session).await.unwrap();
    assert_eq!(
        items.iter().map(|i| i.title.as_str()).collect::<Vec<_>>(),
        ["Draft", "Review"]
    );

    // The first request carried the instructions, the tools and the list.
    let options = client.options(0);
    assert!(options.tools.iter().any(|t| t.name == "todos_add"));
    let first = client.received(0);
    assert!(first[0].contains("## Todo Items"));
    assert!(any_contains(&first, "### Current todo list\n- none yet"));

    // The next run sees the stored items.
    agent
        .run(vec![Message::user("status?")], Some(&mut session))
        .await
        .unwrap();
    assert!(any_contains(
        &client.received(2),
        "### Current todo list\n- 1 [open] Draft\n- 2 [open] Review: twice"
    ));
}

#[tokio::test]
async fn loop_keeps_going_while_todos_remain() {
    let client = Mock::new();
    let todos = TodoProvider::new();
    let inner = Arc::new(
        Agent::builder(client.clone())
            .context_provider(Arc::new(todos.clone()))
            .build(),
    );
    let mut session = inner.create_session();
    session.state.insert(
        "todo",
        json!({"items": [{"id": 1, "title": "Ship it", "description": null, "is_complete": false}],
               "next_id": 2}),
    );
    // Pass 1 does nothing; pass 2 completes the item; the loop then stops.
    client.push(ChatResponse::from_text("thinking"));
    client.push(call_response(&[(
        "c1",
        "todos_complete",
        r#"{"items": [{"id": 1, "reason": "shipped"}]}"#,
    )]));
    client.push(ChatResponse::from_text("shipped"));
    let looping = LoopAgent::builder(inner, todos_remaining(&todos))
        .next_message(todos_remaining_message(&todos))
        .inject_progress(false)
        .build()
        .unwrap();
    let response = looping
        .run(vec![Message::user("finish the list")], Some(&mut session))
        .await
        .unwrap();
    assert!(response.text().ends_with("shipped"));
    assert!(todos.remaining(&session).await.unwrap().is_empty());
    // The second pass was driven by the open-todo reminder.
    assert!(client.received(1).last().unwrap().starts_with(
        "You still have 1 open todo item(s) that must be addressed before you can finish:\n- Ship it"
    ));
}

#[tokio::test]
async fn mode_provider_runs_through_an_agent() {
    let client = Mock::new();
    client.push(call_response(&[(
        "c1",
        "mode_set",
        r#"{"mode": "execute"}"#,
    )]));
    client.push(ChatResponse::from_text("switched"));
    let modes = AgentModeProvider::new();
    let agent = Agent::builder(client.clone())
        .context_provider(Arc::new(modes.clone()))
        .build();
    let mut session = agent.create_session();
    agent
        .run(vec![Message::user("go")], Some(&mut session))
        .await
        .unwrap();
    assert_eq!(modes.mode(&session).unwrap(), "execute");
    assert!(client.received(0)[0].contains("You are currently operating in the plan mode."));

    // An external change is announced once, as a user message.
    modes.set_mode(&session, "plan").unwrap();
    agent
        .run(vec![Message::user("next")], Some(&mut session))
        .await
        .unwrap();
    let seen = client.received(2);
    assert!(seen[0].contains("You are currently operating in the plan mode."));
    assert!(any_contains(
        &seen,
        "[Mode changed: The operating mode has been switched from \"execute\" to \"plan\"."
    ));
    agent
        .run(vec![Message::user("again")], Some(&mut session))
        .await
        .unwrap();
    assert!(!any_contains(&client.received(3), "[Mode changed"));
}
