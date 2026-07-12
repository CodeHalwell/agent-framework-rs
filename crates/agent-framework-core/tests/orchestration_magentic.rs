//! Magentic orchestration tests: a scripted `StandardMagenticManager` driving
//! two agents to completion, plus custom-manager stall/replan and round-limit
//! behavior. No network.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use agent_framework_core::prelude::*;
use agent_framework_core::types::ChatResponseUpdate;
use agent_framework_core::workflow::{MagenticContext, MagenticManager, MagenticProgressLedger};
use async_trait::async_trait;
use futures::StreamExt;

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
        _messages: Vec<Message>,
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
        messages: Vec<Message>,
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

fn agent(name: &str, replies: Vec<&str>) -> Arc<dyn SupportsAgentRun> {
    let responses = replies.into_iter().map(ChatResponse::from_text).collect();
    Arc::new(
        Agent::builder(MockClient::new(responses))
            .name(name)
            .build(),
    ) as Arc<dyn SupportsAgentRun>
}

fn ledger_json(
    satisfied: bool,
    progress: bool,
    in_loop: bool,
    next: &str,
    instruction: &str,
) -> String {
    format!(
        r#"{{"is_request_satisfied":{{"reason":"r","answer":{satisfied}}},
"is_in_loop":{{"reason":"r","answer":{in_loop}}},
"is_progress_being_made":{{"reason":"r","answer":{progress}}},
"next_speaker":{{"reason":"r","answer":"{next}"}},
"instruction_or_question":{{"reason":"r","answer":"{instruction}"}}}}"#
    )
}

fn conversation(run: &WorkflowRun) -> Vec<Message> {
    serde_json::from_value(run.last_output().expect("magentic yields output")).unwrap()
}

#[tokio::test]
async fn standard_manager_drives_agents_to_completion() {
    // Manager script: facts, plan, ledger(select coder), ledger(satisfied), final answer.
    let manager_client = MockClient::new(vec![
        ChatResponse::from_text("GIVEN OR VERIFIED FACTS: none"),
        ChatResponse::from_text("PLAN: ask the coder"),
        ChatResponse::from_text(ledger_json(false, true, false, "coder", "write the code")),
        ChatResponse::from_text(ledger_json(true, true, false, "coder", "")),
        ChatResponse::from_text("FINAL ANSWER: the sum is 42"),
    ]);
    let manager_agent =
        Arc::new(Agent::builder(manager_client).name("mgr").build()) as Arc<dyn SupportsAgentRun>;
    let manager = StandardMagenticManager::new(manager_agent).max_round_count(10);

    let coder = agent("coder", vec!["def solve(): return 42"]);
    let researcher = agent("researcher", vec!["not needed"]);

    let workflow = MagenticBuilder::new()
        .participant("coder", coder)
        .participant("researcher", researcher)
        .standard_manager(manager)
        .build()
        .unwrap();

    let run = workflow.run("compute the answer").await.unwrap();
    let conv = conversation(&run);
    let texts: Vec<String> = conv.iter().map(Message::text).collect();
    assert!(
        texts
            .iter()
            .any(|t| t.contains("FINAL ANSWER: the sum is 42")),
        "final answer synthesized: {texts:?}"
    );
}

/// A custom, fully-scripted manager for exercising stall/replan and round-limit
/// paths deterministically (also demonstrates the `MagenticManager` trait).
struct ScriptedManager {
    ledgers: Mutex<Vec<MagenticProgressLedger>>,
    fixed_ledger: Option<MagenticProgressLedger>,
    plan_calls: Arc<AtomicUsize>,
    replan_calls: Arc<AtomicUsize>,
    final_calls: Arc<AtomicUsize>,
    max_stall: usize,
    max_rounds: Option<usize>,
    max_resets: Option<usize>,
}

fn ledger(satisfied: bool, progress: bool, in_loop: bool, next: &str) -> MagenticProgressLedger {
    serde_json::from_str(&ledger_json(satisfied, progress, in_loop, next, "go")).unwrap()
}

#[async_trait]
impl MagenticManager for ScriptedManager {
    async fn plan(&self, _context: &MagenticContext) -> Result<Message> {
        self.plan_calls.fetch_add(1, Ordering::SeqCst);
        Ok(Message::assistant("PLAN"))
    }

    async fn replan(&self, _context: &MagenticContext) -> Result<Message> {
        self.replan_calls.fetch_add(1, Ordering::SeqCst);
        Ok(Message::assistant("REPLAN"))
    }

    async fn create_progress_ledger(
        &self,
        _context: &MagenticContext,
    ) -> Result<MagenticProgressLedger> {
        if let Some(fixed) = &self.fixed_ledger {
            return Ok(fixed.clone());
        }
        let mut ledgers = self.ledgers.lock().unwrap();
        if ledgers.is_empty() {
            Ok(ledger(true, true, false, "coder"))
        } else {
            Ok(ledgers.remove(0))
        }
    }

    async fn prepare_final_answer(&self, _context: &MagenticContext) -> Result<Message> {
        self.final_calls.fetch_add(1, Ordering::SeqCst);
        Ok(Message::assistant("FINAL"))
    }

    fn max_stall_count(&self) -> usize {
        self.max_stall
    }
    fn max_reset_count(&self) -> Option<usize> {
        self.max_resets
    }
    fn max_round_count(&self) -> Option<usize> {
        self.max_rounds
    }
}

#[tokio::test]
async fn stall_triggers_replan_then_completes() {
    let replan_calls = Arc::new(AtomicUsize::new(0));
    let manager = ScriptedManager {
        // Two no-progress rounds (stall) then satisfied.
        ledgers: Mutex::new(vec![
            ledger(false, false, false, "coder"),
            ledger(false, false, false, "coder"),
            ledger(true, true, false, "coder"),
        ]),
        fixed_ledger: None,
        plan_calls: Arc::new(AtomicUsize::new(0)),
        replan_calls: replan_calls.clone(),
        final_calls: Arc::new(AtomicUsize::new(0)),
        max_stall: 1,
        max_rounds: Some(20),
        max_resets: Some(5),
    };

    let coder = agent("coder", vec!["partial work"]);
    let workflow = MagenticBuilder::new()
        .participant("coder", coder)
        .manager(Arc::new(manager))
        .build()
        .unwrap();

    let run = workflow.run("hard task").await.unwrap();
    let conv = conversation(&run);
    assert_eq!(
        replan_calls.load(Ordering::SeqCst),
        1,
        "one replan on stall"
    );
    assert!(
        conv.iter().any(|m| m.text().contains("FINAL")),
        "final answer after replan: {conv:?}"
    );
}

#[tokio::test]
async fn round_limit_yields_partial_not_final() {
    let final_calls = Arc::new(AtomicUsize::new(0));
    let manager = ScriptedManager {
        ledgers: Mutex::new(Vec::new()),
        // Never satisfied, always route to coder.
        fixed_ledger: Some(ledger(false, true, false, "coder")),
        plan_calls: Arc::new(AtomicUsize::new(0)),
        replan_calls: Arc::new(AtomicUsize::new(0)),
        final_calls: final_calls.clone(),
        max_stall: 5,
        max_rounds: Some(2),
        max_resets: None,
    };

    let coder = agent("coder", vec!["turn-1", "turn-2"]);
    let workflow = MagenticBuilder::new()
        .participant("coder", coder)
        .manager(Arc::new(manager))
        .build()
        .unwrap();

    let run = workflow.run("endless task").await.unwrap();
    let conv = conversation(&run);
    assert_eq!(run.state(), WorkflowRunState::Idle);
    assert_eq!(
        final_calls.load(Ordering::SeqCst),
        0,
        "round-limit exhaustion yields a partial result, not a synthesized final answer"
    );
    assert!(!conv.is_empty(), "a partial result is yielded");
}

/// `intermediate_output_from` demotes the orchestrator's single final yield
/// (the synthesized final answer) from the workflow's terminal output to a
/// non-terminal `Intermediate` event — useful when Magentic is composed as
/// one stage of a larger pipeline. See [`MagenticBuilder::output_from`] docs
/// for why this is whole-orchestrator-granular rather than per-participant
/// (Magentic compiles to a single executor).
#[tokio::test]
async fn intermediate_output_from_demotes_final_yield() {
    let manager = ScriptedManager {
        ledgers: Mutex::new(Vec::new()), // immediately satisfied
        fixed_ledger: None,
        plan_calls: Arc::new(AtomicUsize::new(0)),
        replan_calls: Arc::new(AtomicUsize::new(0)),
        final_calls: Arc::new(AtomicUsize::new(0)),
        max_stall: 5,
        max_rounds: Some(5),
        max_resets: Some(5),
    };

    let coder = agent("coder", vec!["turn-1"]);
    let workflow = MagenticBuilder::new()
        .participant("coder", coder)
        .manager(Arc::new(manager))
        .intermediate_output_from(["coder"])
        .build()
        .unwrap();

    let run = workflow.run("do something").await.unwrap();

    assert!(
        run.last_output().is_none(),
        "no terminal output should be recorded once demoted to Intermediate"
    );
    let intermediate = run
        .events()
        .iter()
        .filter(|e| matches!(e, WorkflowEvent::Intermediate { .. }))
        .count();
    assert_eq!(
        intermediate, 1,
        "the sole yield became a non-terminal event"
    );
    let output_events = run
        .events()
        .iter()
        .filter(|e| matches!(e, WorkflowEvent::Output { .. }))
        .count();
    assert_eq!(output_events, 0);
}

/// Unknown participant names are rejected at build time.
#[tokio::test]
async fn output_from_rejects_unknown_participant() {
    let coder = agent("coder", vec!["turn-1"]);
    let manager = ScriptedManager {
        ledgers: Mutex::new(Vec::new()),
        fixed_ledger: None,
        plan_calls: Arc::new(AtomicUsize::new(0)),
        replan_calls: Arc::new(AtomicUsize::new(0)),
        final_calls: Arc::new(AtomicUsize::new(0)),
        max_stall: 5,
        max_rounds: Some(5),
        max_resets: Some(5),
    };

    let err = match MagenticBuilder::new()
        .participant("coder", coder)
        .manager(Arc::new(manager))
        .output_from(["nobody"])
        .build()
    {
        Ok(_) => panic!("expected an error"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("nobody"));
}
