//! Magentic plan review (human-in-the-loop): with
//! `MagenticBuilder::with_plan_review()`, the run pauses right after the
//! manager produces its initial plan. The pending request carries a
//! `MagenticPlanReviewRequest` (task / facts / plan / round); you answer with
//! a `MagenticPlanReviewDecision` -- approve, revise-with-comments (manager
//! replans), or revise-with-edited-plan (your text is adopted verbatim).
//!
//! Runs fully offline: the manager here is scripted (no LLM). Swap it for
//! `StandardMagenticManager` over a real agent for production use -- the
//! pause/approve flow is identical.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example magentic_plan_review
//! ```

use std::sync::{Arc, Mutex};

use agent_framework::prelude::*;
use agent_framework::workflow::{
    MagenticContext, MagenticManager, MagenticProgressLedger, MagenticTaskLedger,
};
use async_trait::async_trait;
use serde_json::json;

/// A fully scripted manager: fixed plan, fixed "task is done" progress
/// ledger, fixed final answer. Stands in for `StandardMagenticManager`.
struct ScriptedManager {
    ledger: Mutex<Option<MagenticTaskLedger>>,
}

#[async_trait]
impl MagenticManager for ScriptedManager {
    async fn plan(&self, _context: &MagenticContext) -> Result<Message> {
        let ledger = MagenticTaskLedger {
            facts: Message::assistant("Fact: the release notes live in CHANGELOG.md."),
            plan: Message::assistant("1. Draft the notes. 2. Have the editor review."),
        };
        *self.ledger.lock().unwrap() = Some(ledger);
        Ok(Message::assistant("initial combined ledger"))
    }

    async fn replan(&self, _context: &MagenticContext) -> Result<Message> {
        let ledger = MagenticTaskLedger {
            facts: Message::assistant("Fact: the release notes live in CHANGELOG.md."),
            plan: Message::assistant("1. Draft. 2. Review. 3. Add upgrade warnings."),
        };
        *self.ledger.lock().unwrap() = Some(ledger);
        Ok(Message::assistant("revised combined ledger"))
    }

    async fn create_progress_ledger(
        &self,
        _context: &MagenticContext,
    ) -> Result<MagenticProgressLedger> {
        // Report the request as already satisfied so this demo skips the
        // participant round loop and goes straight to the final answer.
        Ok(serde_json::from_value(json!({
            "is_request_satisfied":    { "reason": "demo", "answer": true },
            "is_in_loop":              { "reason": "demo", "answer": false },
            "is_progress_being_made":  { "reason": "demo", "answer": true },
            "next_speaker":            { "reason": "demo", "answer": "writer" },
            "instruction_or_question": { "reason": "demo", "answer": "" },
        }))?)
    }

    async fn prepare_final_answer(&self, _context: &MagenticContext) -> Result<Message> {
        Ok(Message::assistant("Release notes drafted and reviewed."))
    }

    /// Feeds facts/plan text into the plan-review request.
    fn current_task_ledger(&self) -> Option<MagenticTaskLedger> {
        self.ledger.lock().unwrap().clone()
    }
}

/// Never actually invoked here (the ledger reports "satisfied" immediately),
/// but the builder requires at least one participant.
fn placeholder_participant() -> Arc<dyn Agent> {
    struct Silent;
    #[async_trait]
    impl Agent for Silent {
        async fn run(
            &self,
            _messages: Vec<Message>,
            _thread: Option<&mut AgentThread>,
        ) -> Result<AgentResponse> {
            Ok(AgentResponse::default())
        }
        fn id(&self) -> &str {
            "writer"
        }
    }
    Arc::new(Silent)
}

#[tokio::main]
async fn main() -> Result<()> {
    let workflow = MagenticBuilder::new()
        .participant("writer", placeholder_participant())
        .manager(Arc::new(ScriptedManager {
            ledger: Mutex::new(None),
        }))
        .with_plan_review() // <- opt in to the HITL pause
        .build()?;

    let mut run = workflow.run("Draft release notes for v2.0").await?;
    assert_eq!(run.state(), WorkflowRunState::IdleWithPendingRequests);

    // Inspect the plan the manager wants to execute.
    let pending = run.pending_requests();
    let review: MagenticPlanReviewRequest =
        serde_json::from_value(pending[0].request_data.clone())?;
    println!(
        "plan review round {} for task: {}",
        review.round, review.task
    );
    println!("  facts: {}", review.facts);
    println!("  plan:  {}", review.plan);

    // Ask for a revision with comments -> the manager's replan() runs and a
    // fresh review request opens (round 1).
    let revise = MagenticPlanReviewDecision::revise_with_comments("Also mention upgrade steps.");
    run.send_response(pending[0].request_id.clone(), serde_json::to_value(revise)?)
        .await?;

    let pending = run.pending_requests();
    let review: MagenticPlanReviewRequest =
        serde_json::from_value(pending[0].request_data.clone())?;
    println!("revised plan (round {}): {}", review.round, review.plan);

    // Approve -> the orchestration proceeds to completion.
    let approve = MagenticPlanReviewDecision::approve();
    run.send_response(
        pending[0].request_id.clone(),
        serde_json::to_value(approve)?,
    )
    .await?;
    assert_eq!(run.state(), WorkflowRunState::Idle);

    let conversation: Vec<Message> =
        serde_json::from_value(run.last_output().unwrap_or_default()).unwrap_or_default();
    println!(
        "final: {}",
        conversation.last().map(Message::text).unwrap_or_default()
    );

    Ok(())
}
