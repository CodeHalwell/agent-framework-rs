//! Magentic stall-intervention HITL tests: a scripted `MagenticManager` whose
//! progress ledgers can force a stall, exercising the pause/continue,
//! pause/replan-with-guidance, and pause/abort surfaces through the engine's
//! generic `pending_requests()` / `send_response()` machinery, plus a
//! disabled-by-default auto-replan control. No network.

use std::collections::VecDeque;
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

/// A participant that is never expected to run: the round loop either stalls
/// (before selecting a speaker) or reaches a satisfied ledger straight to the
/// final answer. Still required — `MagenticBuilder::build()` needs one.
fn unused_participant(name: &str) -> Arc<dyn Agent> {
    Arc::new(
        ChatAgent::builder(MockClient::new(vec![ChatResponse::from_text(
            "should not be called",
        )]))
        .name(name)
        .build(),
    ) as Arc<dyn Agent>
}

fn ledger(satisfied: bool, progress: bool, in_loop: bool, next: &str) -> MagenticProgressLedger {
    serde_json::from_str(&format!(
        r#"{{"is_request_satisfied":{{"reason":"r","answer":{satisfied}}},
"is_in_loop":{{"reason":"r","answer":{in_loop}}},
"is_progress_being_made":{{"reason":"r","answer":{progress}}},
"next_speaker":{{"reason":"r","answer":"{next}"}},
"instruction_or_question":{{"reason":"r","answer":"go"}}}}"#
    ))
    .unwrap()
}

fn stalled() -> MagenticProgressLedger {
    // Not satisfied, no forward progress, nominates "coder" as next speaker.
    ledger(false, false, false, "coder")
}

fn satisfied() -> MagenticProgressLedger {
    ledger(true, true, false, "coder")
}

/// A `MagenticManager` with fully scripted progress ledgers, tracking a
/// decomposed task ledger (like `StandardMagenticManager`) so the stall
/// request surfaces real facts/plan text. `create_progress_ledger` records
/// whether the history it was handed carried the human's stall guidance.
struct ScriptedStallManager {
    ledgers: Arc<Mutex<VecDeque<MagenticProgressLedger>>>,
    task_ledger: Mutex<Option<MagenticTaskLedger>>,
    plan_calls: Arc<AtomicUsize>,
    replan_calls: Arc<AtomicUsize>,
    final_calls: Arc<AtomicUsize>,
    saw_guidance: Arc<Mutex<bool>>,
    max_stall: usize,
}

impl ScriptedStallManager {
    fn new(ledgers: Vec<MagenticProgressLedger>, max_stall: usize) -> Self {
        Self {
            ledgers: Arc::new(Mutex::new(ledgers.into())),
            task_ledger: Mutex::new(None),
            plan_calls: Arc::new(AtomicUsize::new(0)),
            replan_calls: Arc::new(AtomicUsize::new(0)),
            final_calls: Arc::new(AtomicUsize::new(0)),
            saw_guidance: Arc::new(Mutex::new(false)),
            max_stall,
        }
    }
}

#[async_trait]
impl MagenticManager for ScriptedStallManager {
    async fn plan(&self, _context: &MagenticContext) -> Result<ChatMessage> {
        self.plan_calls.fetch_add(1, Ordering::SeqCst);
        *self.task_ledger.lock().unwrap() = Some(MagenticTaskLedger {
            facts: ChatMessage::assistant("FACTS v1"),
            plan: ChatMessage::assistant("PLAN v1"),
        });
        Ok(ChatMessage::assistant("combined ledger v1"))
    }

    async fn replan(&self, _context: &MagenticContext) -> Result<ChatMessage> {
        self.replan_calls.fetch_add(1, Ordering::SeqCst);
        *self.task_ledger.lock().unwrap() = Some(MagenticTaskLedger {
            facts: ChatMessage::assistant("FACTS v2"),
            plan: ChatMessage::assistant("PLAN v2"),
        });
        Ok(ChatMessage::assistant("combined ledger v2"))
    }

    async fn create_progress_ledger(
        &self,
        context: &MagenticContext,
    ) -> Result<MagenticProgressLedger> {
        if context
            .chat_history
            .iter()
            .any(|m| m.text().contains("Human guidance to help with stall"))
        {
            *self.saw_guidance.lock().unwrap() = true;
        }
        Ok(self
            .ledgers
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(satisfied))
    }

    async fn prepare_final_answer(&self, _context: &MagenticContext) -> Result<ChatMessage> {
        self.final_calls.fetch_add(1, Ordering::SeqCst);
        Ok(ChatMessage::assistant("FINAL ANSWER"))
    }

    fn max_stall_count(&self) -> usize {
        self.max_stall
    }

    fn current_task_ledger(&self) -> Option<MagenticTaskLedger> {
        self.task_ledger.lock().unwrap().clone()
    }
}

fn conversation(run: &WorkflowRun) -> Vec<ChatMessage> {
    serde_json::from_value(run.last_output().expect("magentic yields output")).unwrap()
}

#[tokio::test]
async fn stall_pauses_with_request_fields() {
    // max_stall = 0, so the first no-progress round trips the threshold.
    let manager = ScriptedStallManager::new(vec![stalled()], 0);
    let final_calls = manager.final_calls.clone();

    let workflow = MagenticBuilder::new()
        .participant("coder", unused_participant("coder"))
        .manager(Arc::new(manager))
        .with_stall_intervention()
        .build()
        .unwrap();

    let run = workflow.run("solve the puzzle").await.unwrap();

    assert_eq!(run.state(), WorkflowRunState::IdleWithPendingRequests);
    assert_eq!(final_calls.load(Ordering::SeqCst), 0, "no final answer yet");

    let pending = run.pending_requests();
    assert_eq!(
        pending.len(),
        1,
        "one stall-intervention request outstanding"
    );

    let request: MagenticStallInterventionRequest =
        serde_json::from_value(pending[0].request_data.clone()).unwrap();
    assert_eq!(request.task, "solve the puzzle");
    assert_eq!(request.reason, "No progress being made");
    assert_eq!(request.stall_count, 1);
    assert_eq!(request.max_stall_count, 0);
    assert_eq!(request.round, 1);
    assert_eq!(request.resets_so_far, 0);
    assert_eq!(request.facts, "FACTS v1");
    assert_eq!(request.plan, "PLAN v1");
    assert_eq!(request.last_agent, "coder");
}

#[tokio::test]
async fn stall_reason_reports_looping() {
    // Progress is being made but the team is looping: reason names the loop.
    let manager = ScriptedStallManager::new(vec![ledger(false, true, true, "coder")], 0);
    let workflow = MagenticBuilder::new()
        .participant("coder", unused_participant("coder"))
        .manager(Arc::new(manager))
        .with_stall_intervention()
        .build()
        .unwrap();

    let run = workflow.run("t").await.unwrap();
    let pending = run.pending_requests();
    let request: MagenticStallInterventionRequest =
        serde_json::from_value(pending[0].request_data.clone()).unwrap();
    assert_eq!(request.reason, "Agents appear to be in a loop");
}

#[tokio::test]
async fn stall_continue_clears_counter_and_proceeds() {
    // Stall on round 1, then a satisfied ledger once resumed.
    let manager = ScriptedStallManager::new(vec![stalled(), satisfied()], 0);
    let replan_calls = manager.replan_calls.clone();
    let final_calls = manager.final_calls.clone();

    let workflow = MagenticBuilder::new()
        .participant("coder", unused_participant("coder"))
        .manager(Arc::new(manager))
        .with_stall_intervention()
        .build()
        .unwrap();

    let mut run = workflow.run("solve the puzzle").await.unwrap();
    assert_eq!(run.state(), WorkflowRunState::IdleWithPendingRequests);
    let id = run.pending_requests()[0].request_id.clone();

    let decision =
        serde_json::to_value(MagenticStallInterventionDecision::continue_as_is()).unwrap();
    run.send_response(id, decision).await.unwrap();

    assert_eq!(run.state(), WorkflowRunState::Idle, "run completes");
    assert_eq!(
        replan_calls.load(Ordering::SeqCst),
        0,
        "continue must not replan"
    );
    assert_eq!(final_calls.load(Ordering::SeqCst), 1);

    let texts: Vec<String> = conversation(&run).iter().map(ChatMessage::text).collect();
    assert!(
        texts.iter().any(|t| t.contains("FINAL ANSWER")),
        "{texts:?}"
    );
}

#[tokio::test]
async fn stall_replan_with_guidance_reaches_manager_history() {
    let manager = ScriptedStallManager::new(vec![stalled(), satisfied()], 0);
    let replan_calls = manager.replan_calls.clone();
    let final_calls = manager.final_calls.clone();
    let saw_guidance = manager.saw_guidance.clone();

    let workflow = MagenticBuilder::new()
        .participant("coder", unused_participant("coder"))
        .manager(Arc::new(manager))
        .with_stall_intervention()
        .build()
        .unwrap();

    let mut run = workflow.run("solve the puzzle").await.unwrap();
    let id = run.pending_requests()[0].request_id.clone();

    let decision = serde_json::to_value(MagenticStallInterventionDecision::replan_with_guidance(
        "focus on the edge cases",
    ))
    .unwrap();
    run.send_response(id, decision).await.unwrap();

    assert_eq!(run.state(), WorkflowRunState::Idle);
    assert_eq!(replan_calls.load(Ordering::SeqCst), 1, "replan was forced");
    assert!(
        *saw_guidance.lock().unwrap(),
        "the human's guidance reached the manager's history"
    );
    assert_eq!(final_calls.load(Ordering::SeqCst), 1);

    // The guidance is also visible on the run's emitted messages.
    let guidance_emitted = run.events().iter().any(|e| {
        serde_json::to_value(e)
            .map(|v| v.to_string().contains("Human guidance to help with stall"))
            .unwrap_or(false)
    });
    assert!(
        guidance_emitted,
        "guidance surfaced as an orchestrator event"
    );
}

#[tokio::test]
async fn stall_abort_produces_final_answer() {
    let manager = ScriptedStallManager::new(vec![stalled()], 0);
    let replan_calls = manager.replan_calls.clone();
    let final_calls = manager.final_calls.clone();

    let workflow = MagenticBuilder::new()
        .participant("coder", unused_participant("coder"))
        .manager(Arc::new(manager))
        .with_stall_intervention()
        .build()
        .unwrap();

    let mut run = workflow.run("solve the puzzle").await.unwrap();
    let id = run.pending_requests()[0].request_id.clone();

    let decision = serde_json::to_value(MagenticStallInterventionDecision::abort()).unwrap();
    run.send_response(id, decision).await.unwrap();

    assert_eq!(run.state(), WorkflowRunState::Idle);
    assert_eq!(
        replan_calls.load(Ordering::SeqCst),
        0,
        "abort does not replan"
    );
    assert_eq!(
        final_calls.load(Ordering::SeqCst),
        1,
        "abort synthesizes a final answer"
    );

    let texts: Vec<String> = conversation(&run).iter().map(ChatMessage::text).collect();
    assert!(
        texts.iter().any(|t| t.contains("FINAL ANSWER")),
        "{texts:?}"
    );
}

#[tokio::test]
async fn disabled_stall_intervention_auto_replans() {
    // Without `with_stall_intervention`, a stall silently auto-replans (the
    // pre-existing behavior) rather than pausing.
    let manager = ScriptedStallManager::new(vec![stalled(), satisfied()], 0);
    let replan_calls = manager.replan_calls.clone();
    let final_calls = manager.final_calls.clone();

    let workflow = MagenticBuilder::new()
        .participant("coder", unused_participant("coder"))
        .manager(Arc::new(manager))
        .build()
        .unwrap();

    let run = workflow.run("solve the puzzle").await.unwrap();

    assert_eq!(
        run.state(),
        WorkflowRunState::Idle,
        "no pause: the run runs to completion in one shot"
    );
    assert!(run.pending_requests().is_empty(), "no HITL request emitted");
    assert_eq!(
        replan_calls.load(Ordering::SeqCst),
        1,
        "the stall auto-replanned"
    );
    assert_eq!(final_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn stall_decision_serde_shapes() {
    assert_eq!(
        serde_json::to_value(MagenticStallInterventionDecision::continue_as_is()).unwrap(),
        json!({ "decision": "continue" })
    );
    assert_eq!(
        serde_json::to_value(MagenticStallInterventionDecision::replan()).unwrap(),
        json!({ "decision": "replan" })
    );
    assert_eq!(
        serde_json::to_value(MagenticStallInterventionDecision::replan_with_guidance("g")).unwrap(),
        json!({ "decision": "replan", "guidance": "g" })
    );
    assert_eq!(
        serde_json::to_value(MagenticStallInterventionDecision::abort()).unwrap(),
        json!({ "decision": "abort" })
    );
}
