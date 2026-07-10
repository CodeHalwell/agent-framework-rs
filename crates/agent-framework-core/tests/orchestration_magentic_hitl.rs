//! Magentic plan-review HITL tests: a scripted `MagenticManager` exercising
//! the pause/approve and pause/revise/approve loops through the engine's
//! generic `pending_requests()` / `send_response()` machinery. No network.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use agent_framework_core::prelude::*;
use agent_framework_core::types::ChatResponseUpdate;
use agent_framework_core::workflow::{
    MagenticContext, MagenticManager, MagenticProgressLedger, MagenticTaskLedger,
};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::json;

#[derive(Clone)]
struct MockClient {
    responses: Arc<Mutex<Vec<ChatResponse>>>,
}

impl MockClient {
    fn new(responses: Vec<ChatResponse>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses)),
        }
    }
}

#[async_trait]
impl ChatClient for MockClient {
    async fn get_response(
        &self,
        _messages: Vec<ChatMessage>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        let mut resps = self.responses.lock().unwrap();
        if resps.is_empty() {
            Ok(ChatResponse::from_text("(no more scripted responses)"))
        } else {
            Ok(resps.remove(0))
        }
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<ChatMessage>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let resp = self.get_response(messages, options).await?;
        let updates: Vec<Result<ChatResponseUpdate>> = resp
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

/// A participant agent that is never actually expected to run in these
/// tests: every scripted progress ledger reports the request as already
/// satisfied, so the round loop goes straight to `prepare_final_answer`
/// without selecting a next speaker. Still required because
/// `MagenticBuilder::build()` needs at least one participant.
fn unused_participant(name: &str) -> Arc<dyn Agent> {
    Arc::new(
        ChatAgent::builder(MockClient::new(vec![ChatResponse::from_text(
            "should not be called",
        )]))
        .name(name)
        .build(),
    ) as Arc<dyn Agent>
}

fn satisfied_ledger() -> MagenticProgressLedger {
    serde_json::from_str(
        r#"{"is_request_satisfied":{"reason":"r","answer":true},
"is_in_loop":{"reason":"r","answer":false},
"is_progress_being_made":{"reason":"r","answer":true},
"next_speaker":{"reason":"r","answer":"coder"},
"instruction_or_question":{"reason":"r","answer":""}}"#,
    )
    .unwrap()
}

/// A `MagenticManager` whose `plan`/`replan` are fully scripted and which
/// tracks a decomposed [`MagenticTaskLedger`] the way `StandardMagenticManager`
/// does, so `current_task_ledger()` can feed real facts/plan text into plan
/// review. `replan` inspects the context handed to it and reports whether it
/// saw the human's feedback message, so tests can assert the feedback
/// actually reached the manager (mirrors Python appending
/// `"Human plan feedback: ..."` to `chat_history` before calling `replan`).
struct ScriptedManager {
    task_ledger: Mutex<Option<MagenticTaskLedger>>,
    plan_calls: Arc<AtomicUsize>,
    replan_calls: Arc<AtomicUsize>,
    final_calls: Arc<AtomicUsize>,
    saw_feedback_on_last_replan: Arc<Mutex<bool>>,
}

impl ScriptedManager {
    fn new() -> Self {
        Self {
            task_ledger: Mutex::new(None),
            plan_calls: Arc::new(AtomicUsize::new(0)),
            replan_calls: Arc::new(AtomicUsize::new(0)),
            final_calls: Arc::new(AtomicUsize::new(0)),
            saw_feedback_on_last_replan: Arc::new(Mutex::new(false)),
        }
    }
}

#[async_trait]
impl MagenticManager for ScriptedManager {
    async fn plan(&self, _context: &MagenticContext) -> Result<ChatMessage> {
        self.plan_calls.fetch_add(1, Ordering::SeqCst);
        let facts = ChatMessage::assistant("FACTS v1");
        let plan = ChatMessage::assistant("PLAN v1");
        *self.task_ledger.lock().unwrap() = Some(MagenticTaskLedger {
            facts: facts.clone(),
            plan: plan.clone(),
        });
        Ok(ChatMessage::assistant("combined ledger v1"))
    }

    async fn replan(&self, context: &MagenticContext) -> Result<ChatMessage> {
        self.replan_calls.fetch_add(1, Ordering::SeqCst);
        let saw_feedback = context
            .chat_history
            .iter()
            .any(|m| m.text().contains("Human plan feedback"));
        *self.saw_feedback_on_last_replan.lock().unwrap() = saw_feedback;

        let facts = ChatMessage::assistant("FACTS v1");
        let plan = ChatMessage::assistant("REVISED PLAN");
        *self.task_ledger.lock().unwrap() = Some(MagenticTaskLedger {
            facts: facts.clone(),
            plan: plan.clone(),
        });
        Ok(ChatMessage::assistant("combined ledger v2"))
    }

    async fn create_progress_ledger(
        &self,
        _context: &MagenticContext,
    ) -> Result<MagenticProgressLedger> {
        Ok(satisfied_ledger())
    }

    async fn prepare_final_answer(&self, _context: &MagenticContext) -> Result<ChatMessage> {
        self.final_calls.fetch_add(1, Ordering::SeqCst);
        Ok(ChatMessage::assistant("FINAL ANSWER"))
    }

    fn current_task_ledger(&self) -> Option<MagenticTaskLedger> {
        self.task_ledger.lock().unwrap().clone()
    }
}

fn conversation(run: &WorkflowRun) -> Vec<ChatMessage> {
    serde_json::from_value(run.last_output().expect("magentic yields output")).unwrap()
}

#[tokio::test]
async fn plan_review_pauses_with_request_then_approve_completes() {
    let manager = ScriptedManager::new();
    let plan_calls = manager.plan_calls.clone();
    let replan_calls = manager.replan_calls.clone();
    let final_calls = manager.final_calls.clone();

    let workflow = MagenticBuilder::new()
        .participant("coder", unused_participant("coder"))
        .manager(Arc::new(manager))
        .with_plan_review()
        .build()
        .unwrap();

    let mut run = workflow.run("solve the puzzle").await.unwrap();

    // The run must pause immediately after the initial plan, before ever
    // touching the round loop.
    assert_eq!(run.state(), WorkflowRunState::IdleWithPendingRequests);
    assert_eq!(plan_calls.load(Ordering::SeqCst), 1, "initial plan() ran");
    assert_eq!(
        final_calls.load(Ordering::SeqCst),
        0,
        "round loop hasn't run yet"
    );

    let pending = run.pending_requests();
    assert_eq!(pending.len(), 1, "one plan-review request outstanding");

    let request: MagenticPlanReviewRequest =
        serde_json::from_value(pending[0].request_data.clone()).unwrap();
    assert_eq!(request.task, "solve the puzzle");
    assert_eq!(request.facts, "FACTS v1");
    assert_eq!(request.plan, "PLAN v1");
    assert_eq!(request.round, 0, "first request is round 0");

    // Approve outright: no edits, no comments -> no replan call, proceed.
    let decision = serde_json::to_value(MagenticPlanReviewDecision::approve()).unwrap();
    run.send_response(pending[0].request_id.clone(), decision)
        .await
        .unwrap();

    assert_eq!(run.state(), WorkflowRunState::Idle, "run completes");
    assert_eq!(
        replan_calls.load(Ordering::SeqCst),
        0,
        "a bare approve must not call replan"
    );
    assert_eq!(final_calls.load(Ordering::SeqCst), 1);

    let conv = conversation(&run);
    let texts: Vec<String> = conv.iter().map(ChatMessage::text).collect();
    assert!(
        texts.iter().any(|t| t.contains("FINAL ANSWER")),
        "final answer synthesized: {texts:?}"
    );
}

#[tokio::test]
async fn plan_review_revise_then_approve_completes() {
    let manager = ScriptedManager::new();
    let plan_calls = manager.plan_calls.clone();
    let replan_calls = manager.replan_calls.clone();
    let final_calls = manager.final_calls.clone();
    let saw_feedback = manager.saw_feedback_on_last_replan.clone();

    let workflow = MagenticBuilder::new()
        .participant("coder", unused_participant("coder"))
        .manager(Arc::new(manager))
        .with_plan_review()
        .build()
        .unwrap();

    let mut run = workflow.run("solve the puzzle").await.unwrap();
    let first_id = run.pending_requests()[0].request_id.clone();

    // Revise with feedback (no edited_plan): the manager should replan, and
    // the human's feedback should reach it via chat_history, mirroring
    // Python's "Human plan feedback: ..." message.
    let decision = serde_json::to_value(MagenticPlanReviewDecision::revise_with_comments(
        "please add tests",
    ))
    .unwrap();
    run.send_response(first_id, decision).await.unwrap();

    assert_eq!(
        run.state(),
        WorkflowRunState::IdleWithPendingRequests,
        "revise re-opens plan review instead of proceeding"
    );
    assert_eq!(
        replan_calls.load(Ordering::SeqCst),
        1,
        "revise called replan"
    );
    assert!(
        *saw_feedback.lock().unwrap(),
        "the human's feedback reached the manager's replan call"
    );

    let pending = run.pending_requests();
    assert_eq!(pending.len(), 1);
    let second_request: MagenticPlanReviewRequest =
        serde_json::from_value(pending[0].request_data.clone()).unwrap();
    assert_eq!(second_request.round, 1, "round advanced after a revise");
    assert_eq!(
        second_request.plan, "REVISED PLAN",
        "the revised plan reached the re-sent ledger"
    );
    assert_eq!(
        second_request.facts, "FACTS v1",
        "facts carried over unchanged"
    );

    // Now approve the revised plan.
    let approve = serde_json::to_value(MagenticPlanReviewDecision::approve()).unwrap();
    run.send_response(pending[0].request_id.clone(), approve)
        .await
        .unwrap();

    assert_eq!(run.state(), WorkflowRunState::Idle, "run completes");
    assert_eq!(plan_calls.load(Ordering::SeqCst), 1, "plan() only ran once");
    assert_eq!(
        replan_calls.load(Ordering::SeqCst),
        1,
        "no extra replan on approve"
    );
    assert_eq!(final_calls.load(Ordering::SeqCst), 1);

    let conv = conversation(&run);
    let texts: Vec<String> = conv.iter().map(ChatMessage::text).collect();
    assert!(
        texts.iter().any(|t| t.contains("FINAL ANSWER")),
        "final answer synthesized after revise+approve: {texts:?}"
    );
}

#[tokio::test]
async fn plan_review_edited_plan_skips_llm_and_is_reflected_immediately() {
    let manager = ScriptedManager::new();
    let replan_calls = manager.replan_calls.clone();

    let workflow = MagenticBuilder::new()
        .participant("coder", unused_participant("coder"))
        .manager(Arc::new(manager))
        .with_plan_review()
        .build()
        .unwrap();

    let mut run = workflow.run("solve the puzzle").await.unwrap();
    let first_id = run.pending_requests()[0].request_id.clone();

    // Revise with a directly-edited plan: no LLM call, just adopt the text
    // and re-ask for approval.
    let decision = serde_json::to_value(MagenticPlanReviewDecision::revise_with_edited_plan(
        "HUMAN-EDITED PLAN",
    ))
    .unwrap();
    run.send_response(first_id, decision).await.unwrap();

    assert_eq!(run.state(), WorkflowRunState::IdleWithPendingRequests);
    assert_eq!(
        replan_calls.load(Ordering::SeqCst),
        0,
        "a direct plan edit must not call replan"
    );

    let pending = run.pending_requests();
    let request: MagenticPlanReviewRequest =
        serde_json::from_value(pending[0].request_data.clone()).unwrap();
    assert_eq!(request.plan, "HUMAN-EDITED PLAN");
    assert_eq!(request.round, 1);

    let approve = serde_json::to_value(MagenticPlanReviewDecision::approve()).unwrap();
    run.send_response(pending[0].request_id.clone(), approve)
        .await
        .unwrap();
    assert_eq!(run.state(), WorkflowRunState::Idle);
}

#[tokio::test]
async fn plan_review_exceeds_max_rounds_force_proceeds() {
    let manager = ScriptedManager::new();
    let final_calls = manager.final_calls.clone();

    let workflow = MagenticBuilder::new()
        .participant("coder", unused_participant("coder"))
        .manager(Arc::new(manager))
        .with_plan_review()
        .max_plan_review_rounds(1)
        .build()
        .unwrap();

    let mut run = workflow.run("solve the puzzle").await.unwrap();

    // Round 1: within the limit, re-opens review.
    let first_id = run.pending_requests()[0].request_id.clone();
    let revise =
        serde_json::to_value(MagenticPlanReviewDecision::revise_with_comments("more")).unwrap();
    run.send_response(first_id, revise.clone()).await.unwrap();
    assert_eq!(run.state(), WorkflowRunState::IdleWithPendingRequests);

    // Round 2: exceeds max_plan_review_rounds(1) -> force-proceed instead of
    // opening a third review request.
    let second_id = run.pending_requests()[0].request_id.clone();
    run.send_response(second_id, revise).await.unwrap();

    assert_eq!(
        run.state(),
        WorkflowRunState::Idle,
        "exceeding the round limit forces completion instead of another pause"
    );
    assert!(run.pending_requests().is_empty());
    assert_eq!(final_calls.load(Ordering::SeqCst), 1);

    let conv = conversation(&run);
    let texts: Vec<String> = conv.iter().map(ChatMessage::text).collect();
    assert!(
        texts.iter().any(|t| t.contains("FINAL ANSWER")),
        "still reaches a final answer: {texts:?}"
    );
}

#[test]
fn plan_review_decision_serde_shapes() {
    let approve = MagenticPlanReviewDecision::approve();
    assert_eq!(
        serde_json::to_value(&approve).unwrap(),
        json!({"decision": "approve"})
    );

    let revise = MagenticPlanReviewDecision::revise_with_comments("needs more detail");
    assert_eq!(
        serde_json::to_value(&revise).unwrap(),
        json!({"decision": "revise", "comments": "needs more detail"})
    );

    let round_trip: MagenticPlanReviewDecision =
        serde_json::from_value(json!({"decision": "approve", "edited_plan": "new text"})).unwrap();
    matches!(
        round_trip,
        MagenticPlanReviewDecision::Approve { edited_plan: Some(ref t), .. } if t == "new text"
    );

    let request = MagenticPlanReviewRequest {
        task: "t".into(),
        facts: "f".into(),
        plan: "p".into(),
        round: 3,
    };
    let value = serde_json::to_value(&request).unwrap();
    let back: MagenticPlanReviewRequest = serde_json::from_value(value).unwrap();
    assert_eq!(back.round, 3);
}
