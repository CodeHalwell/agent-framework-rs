//! Magentic stall intervention (human-in-the-loop): with
//! `MagenticBuilder::with_stall_intervention()`, the orchestration pauses when
//! the manager's progress ledger reports a stall (no progress / agents
//! looping) instead of silently auto-replanning. The pending request carries a
//! `MagenticStallInterventionRequest` (task / facts / plan / reason /
//! stall_count); you answer with a `MagenticStallInterventionDecision` --
//! continue as-is, replan (optionally with guidance fed to the manager), or
//! abort.
//!
//! Runs fully offline: the manager is scripted (no LLM).
//!
//! ```bash
//! cargo run -p agent-framework-examples --example magentic_stall_intervention
//! ```

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use agent_framework::prelude::*;
use agent_framework::workflow::{
    MagenticContext, MagenticManager, MagenticProgressLedger, MagenticTaskLedger,
};
use async_trait::async_trait;
use serde_json::json;

fn ledger(satisfied: bool, progress: bool) -> MagenticProgressLedger {
    serde_json::from_value(json!({
        "is_request_satisfied":    { "reason": "demo", "answer": satisfied },
        "is_in_loop":              { "reason": "demo", "answer": false },
        "is_progress_being_made":  { "reason": "demo", "answer": progress },
        "next_speaker":            { "reason": "demo", "answer": "worker" },
        "instruction_or_question": { "reason": "demo", "answer": "" },
    }))
    .unwrap()
}

/// Scripted manager: first progress ledger reports a stall, later ones report
/// success. `max_stall_count() = 0` makes the very first stall trip the pause.
struct ScriptedManager {
    ledgers: Mutex<VecDeque<MagenticProgressLedger>>,
    task_ledger: Mutex<Option<MagenticTaskLedger>>,
}

#[async_trait]
impl MagenticManager for ScriptedManager {
    async fn plan(&self, _context: &MagenticContext) -> Result<Message> {
        *self.task_ledger.lock().unwrap() = Some(MagenticTaskLedger {
            facts: Message::assistant("Fact: the dataset lives in data/."),
            plan: Message::assistant("1. Load data. 2. Compute stats."),
        });
        Ok(Message::assistant("combined ledger"))
    }

    async fn replan(&self, _context: &MagenticContext) -> Result<Message> {
        println!("  manager replanning after human intervention");
        Ok(Message::assistant("revised combined ledger"))
    }

    async fn create_progress_ledger(
        &self,
        context: &MagenticContext,
    ) -> Result<MagenticProgressLedger> {
        // The human's guidance lands in chat_history (as a "Human guidance to
        // help with stall: ..." message) for every manager call after the
        // intervention, so subsequent ledger evaluations can act on it.
        let guided = context
            .chat_history
            .iter()
            .any(|m| m.text().contains("Human guidance"));
        println!("  progress check (human guidance in history: {guided})");
        Ok(self
            .ledgers
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| ledger(true, true)))
    }

    async fn prepare_final_answer(&self, _context: &MagenticContext) -> Result<Message> {
        Ok(Message::assistant("Stats computed successfully."))
    }

    fn max_stall_count(&self) -> usize {
        0 // pause on the very first stalled round
    }

    fn current_task_ledger(&self) -> Option<MagenticTaskLedger> {
        self.task_ledger.lock().unwrap().clone()
    }
}

/// Never invoked (the demo stalls before any speaker turn, then finishes),
/// but the builder requires a participant.
fn placeholder() -> Arc<dyn SupportsAgentRun> {
    struct Silent;
    #[async_trait]
    impl SupportsAgentRun for Silent {
        async fn run(
            &self,
            _messages: Vec<Message>,
            _thread: Option<&mut AgentThread>,
        ) -> Result<AgentResponse> {
            Ok(AgentResponse::default())
        }
        fn id(&self) -> &str {
            "worker"
        }
    }
    Arc::new(Silent)
}

#[tokio::main]
async fn main() -> Result<()> {
    let manager = ScriptedManager {
        // Stalled first round; satisfied after the human weighs in.
        ledgers: Mutex::new(VecDeque::from([ledger(false, false), ledger(true, true)])),
        task_ledger: Mutex::new(None),
    };

    let workflow = MagenticBuilder::new()
        .participant("worker", placeholder())
        .manager(Arc::new(manager))
        .with_stall_intervention() // <- opt in to the HITL pause
        .build()?;

    let mut run = workflow.run("Analyze the quarterly dataset").await?;
    assert_eq!(run.state(), WorkflowRunState::IdleWithPendingRequests);

    let pending = run.pending_requests();
    let request: MagenticStallInterventionRequest =
        serde_json::from_value(pending[0].request_data.clone())?;
    println!(
        "stalled (count {}): {}\n  plan under review: {}",
        request.stall_count, request.reason, request.plan
    );

    // Ask the manager to replan with guidance. The alternatives:
    // `continue_as_is()` (push on unchanged) and `abort()` (stop the run).
    let decision =
        MagenticStallInterventionDecision::replan_with_guidance("Try sampling the data first.");
    run.send_response(
        pending[0].request_id.clone(),
        serde_json::to_value(decision)?,
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
