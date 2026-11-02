//! The harness background-agents provider (`experimental-harness`).
//!
//! Ported from upstream `test_harness_background_agents.py`.
#![cfg(feature = "experimental-harness")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use agent_framework_core::agent::SupportsAgentRun;
use agent_framework_core::error::{Error, Result};
use agent_framework_core::harness::*;
use agent_framework_core::memory::{ContextProvider, SessionContext};
use agent_framework_core::prelude::*;
use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::Semaphore;

/// An agent that answers `"{name}: {input}"` once a permit is released
/// (immediately with `gate: None`), or fails on input `"fail"`.
struct Worker {
    name: String,
    description: Option<String>,
    gate: Option<Arc<Semaphore>>,
    runs: AtomicUsize,
}

impl Worker {
    fn new(name: &str) -> Arc<Self> {
        Arc::new(Self {
            name: name.into(),
            description: Some(format!("{name} does work")),
            gate: None,
            runs: AtomicUsize::new(0),
        })
    }

    fn gated(name: &str, gate: Arc<Semaphore>) -> Arc<Self> {
        Arc::new(Self {
            name: name.into(),
            description: None,
            gate: Some(gate),
            runs: AtomicUsize::new(0),
        })
    }
}

#[async_trait]
impl SupportsAgentRun for Worker {
    async fn run(
        &self,
        messages: Vec<Message>,
        session: Option<&mut AgentSession>,
    ) -> Result<AgentResponse> {
        let n = self.runs.fetch_add(1, Ordering::SeqCst) + 1;
        if let Some(gate) = &self.gate {
            gate.acquire().await.unwrap().forget();
        }
        let input = messages.last().map(|m| m.text()).unwrap_or_default();
        if input == "fail" {
            return Err(Error::Other("boom".into()));
        }
        // Prove the sub-session persists across a continuation.
        let session = session.expect("background tasks run in a session");
        let seen = session
            .state
            .get("seen")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        session.state.insert("seen", json!(seen + 1));
        Ok(AgentResponse {
            messages: vec![Message::assistant(format!(
                "{}: {input} (run {n}, seen {seen})",
                self.name
            ))],
            ..Default::default()
        })
    }

    fn id(&self) -> &str {
        &self.name
    }

    fn name(&self) -> Option<&str> {
        Some(&self.name)
    }

    fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }
}

async fn prepare(provider: &BackgroundAgentsProvider, session: &AgentSession) -> SessionContext {
    let mut ctx = SessionContext::new(vec![Message::user("hi")]);
    ctx.session = Some(session.clone());
    provider.before_run(&mut ctx).await.unwrap();
    ctx
}

async fn call(ctx: &SessionContext, name: &str, args: Value) -> String {
    ctx.tools
        .iter()
        .find(|t| t.name == name)
        .unwrap_or_else(|| panic!("no tool {name}"))
        .executor
        .as_ref()
        .unwrap()
        .invoke(args)
        .await
        .unwrap()
        .as_str()
        .unwrap()
        .to_string()
}

#[test]
fn validates_agents_and_timeout() {
    assert!(BackgroundAgentsProvider::new(Vec::<Arc<dyn SupportsAgentRun>>::new()).is_err());
    let dup: Vec<Arc<dyn SupportsAgentRun>> = vec![Worker::new("A"), Worker::new("a")];
    let err = BackgroundAgentsProvider::new(dup).unwrap_err();
    assert!(
        err.to_string().contains("Duplicate background agent name"),
        "{err}"
    );
    let one: Vec<Arc<dyn SupportsAgentRun>> = vec![Worker::new("a")];
    assert!(BackgroundAgentsProvider::builder(one)
        .wait_timeout(Duration::ZERO)
        .build()
        .is_err());
}

#[tokio::test]
async fn instructions_list_agents_and_tools_are_advertised() {
    let provider = BackgroundAgentsProvider::new(vec![
        Worker::new("researcher") as Arc<dyn SupportsAgentRun>,
        Worker::gated("writer", Arc::new(Semaphore::new(0))),
    ])
    .unwrap();
    let session = AgentSession::new();
    let ctx = prepare(&provider, &session).await;
    let instructions = ctx.instructions.clone().unwrap();
    assert!(instructions
        .contains("Available background agents:\n- researcher: researcher does work\n- writer"));
    let names: Vec<_> = ctx.tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "background_agents_start_task",
            "background_agents_wait_for_first_completion",
            "background_agents_get_task_results",
            "background_agents_get_all_tasks",
            "background_agents_continue_task",
            "background_agents_clear_completed_task",
        ]
    );
    assert!(ctx.messages.is_empty(), "no task list without tasks");
}

#[tokio::test]
async fn start_wait_results_continue_and_clear() {
    let worker = Worker::new("researcher");
    let provider =
        BackgroundAgentsProvider::new(vec![worker.clone() as Arc<dyn SupportsAgentRun>]).unwrap();
    let session = AgentSession::new();
    let ctx = prepare(&provider, &session).await;

    assert_eq!(
        call(
            &ctx,
            "background_agents_start_task",
            json!({"agent_name": "Researcher", "input": "find x", "description": "look up x"})
        )
        .await,
        "Background task 1 started on agent 'Researcher'."
    );
    assert_eq!(
        call(
            &ctx,
            "background_agents_wait_for_first_completion",
            json!({"task_ids": [1]})
        )
        .await,
        "Task 1 finished with status: completed."
    );
    assert_eq!(
        call(
            &ctx,
            "background_agents_get_task_results",
            json!({"task_id": 1})
        )
        .await,
        "researcher: find x (run 1, seen 0)"
    );
    assert_eq!(
        call(
            &ctx,
            "background_agents_wait_for_first_completion",
            json!({"task_ids": [1]})
        )
        .await,
        "Task 1 is not running; current status: completed."
    );

    // Continuing reuses the task's own session.
    assert_eq!(
        call(
            &ctx,
            "background_agents_continue_task",
            json!({"task_id": 1, "text": "more"})
        )
        .await,
        "Task 1 continued with new input."
    );
    call(
        &ctx,
        "background_agents_wait_for_first_completion",
        json!({"task_ids": [1]}),
    )
    .await;
    assert_eq!(
        call(
            &ctx,
            "background_agents_get_task_results",
            json!({"task_id": 1})
        )
        .await,
        "researcher: more (run 2, seen 1)"
    );

    assert_eq!(
        call(&ctx, "background_agents_get_all_tasks", json!({})).await,
        "Tasks:\n- Task 1 [completed] (Researcher): look up x"
    );
    // The next turn lists the tasks for the model.
    let next = prepare(&provider, &session).await;
    assert_eq!(
        next.messages[0].text(),
        "### Current background tasks\n- Task 1 [completed] (Researcher): look up x"
    );

    assert_eq!(
        call(
            &ctx,
            "background_agents_clear_completed_task",
            json!({"task_id": 1})
        )
        .await,
        "Task 1 cleared."
    );
    assert_eq!(
        call(&ctx, "background_agents_get_all_tasks", json!({})).await,
        "No tasks."
    );
    assert_eq!(
        call(
            &ctx,
            "background_agents_start_task",
            json!({"agent_name": "nobody", "input": "x", "description": "d"})
        )
        .await,
        "Error: No background agent found with name 'nobody'. Available agents: researcher"
    );
}

#[tokio::test]
async fn failures_timeouts_and_running_guards() {
    let gate = Arc::new(Semaphore::new(0));
    let provider = BackgroundAgentsProvider::builder(vec![
        Worker::gated("slow", gate.clone()) as Arc<dyn SupportsAgentRun>
    ])
    .wait_timeout(Duration::from_millis(50))
    .build()
    .unwrap();
    let session = AgentSession::new();
    let ctx = prepare(&provider, &session).await;

    call(
        &ctx,
        "background_agents_start_task",
        json!({"agent_name": "slow", "input": "fail", "description": "will fail"}),
    )
    .await;
    let waited = call(
        &ctx,
        "background_agents_wait_for_first_completion",
        json!({"task_ids": [1]}),
    )
    .await;
    assert!(
        waited.starts_with("No background task completed within 0 seconds."),
        "{waited}"
    );
    assert_eq!(
        call(
            &ctx,
            "background_agents_get_task_results",
            json!({"task_id": 1})
        )
        .await,
        "Task 1 is still running."
    );
    assert_eq!(
        call(
            &ctx,
            "background_agents_continue_task",
            json!({"task_id": 1, "text": "x"})
        )
        .await,
        "Error: Task 1 is still running. Wait for it to complete before continuing."
    );
    assert_eq!(
        call(
            &ctx,
            "background_agents_clear_completed_task",
            json!({"task_id": 1})
        )
        .await,
        "Error: Task 1 is still running. Wait for it to complete before clearing."
    );

    // The loop helpers see the running task.
    assert_eq!(
        provider.persisted_tasks(&session).unwrap()[0].status,
        BackgroundTaskStatus::Running
    );

    gate.add_permits(1);
    assert_eq!(
        call(
            &ctx,
            "background_agents_wait_for_first_completion",
            json!({"task_ids": [1]})
        )
        .await,
        "Task 1 finished with status: failed."
    );
    let failed = call(
        &ctx,
        "background_agents_get_task_results",
        json!({"task_id": 1}),
    )
    .await;
    assert!(
        failed.starts_with("Task failed: ") && failed.contains("boom"),
        "{failed}"
    );
}

#[tokio::test]
async fn tasks_without_a_runtime_are_lost_and_release_cancels() {
    let gate = Arc::new(Semaphore::new(0));
    let agents = || vec![Worker::gated("slow", gate.clone()) as Arc<dyn SupportsAgentRun>];
    let provider = BackgroundAgentsProvider::new(agents()).unwrap();
    let session = AgentSession::new();
    let ctx = prepare(&provider, &session).await;
    call(
        &ctx,
        "background_agents_start_task",
        json!({"agent_name": "slow", "input": "x", "description": "d"}),
    )
    .await;

    // Releasing without cancelling refuses while a task runs.
    assert!(provider
        .release_session(&session, false, None)
        .await
        .is_err());
    provider
        .release_session(&session, true, Some(Duration::from_secs(5)))
        .await
        .unwrap();
    assert_eq!(
        call(
            &ctx,
            "background_agents_start_task",
            json!({"agent_name": "slow", "input": "x", "description": "d"})
        )
        .await,
        "Error: Session is being released; cannot start a new background task."
    );

    // A fresh runtime (as after a restart) no longer knows the task.
    let next = prepare(&provider, &session).await;
    assert_eq!(
        call(
            &next,
            "background_agents_get_task_results",
            json!({"task_id": 1})
        )
        .await,
        "Task state was lost (reference unavailable)."
    );
    assert_eq!(
        call(
            &next,
            "background_agents_continue_task",
            json!({"task_id": 1, "text": "x"})
        )
        .await,
        "Error: Task 1 cannot be continued because its session was lost. Start a new task instead."
    );
}

#[tokio::test]
async fn loop_callbacks_follow_running_tasks() {
    use agent_framework_core::harness::{LoopCallback, LoopContext, LoopDecision};
    let gate = Arc::new(Semaphore::new(0));
    let provider = BackgroundAgentsProvider::new(vec![
        Worker::gated("slow", gate.clone()) as Arc<dyn SupportsAgentRun>
    ])
    .unwrap();
    let session = AgentSession::new();
    let ctx = prepare(&provider, &session).await;
    let loop_ctx = LoopContext {
        iteration: 1,
        last_result: AgentResponse::default(),
        messages: vec![],
        original_messages: vec![],
        session: session.clone(),
        progress: vec![],
        feedback: None,
    };
    let cond = background_tasks_running(&provider);
    let message = background_tasks_running_message(&provider);
    assert_eq!(cond.call(&loop_ctx).await.unwrap(), LoopDecision::stop());
    assert!(message.call(&loop_ctx).await.unwrap().is_none());

    call(
        &ctx,
        "background_agents_start_task",
        json!({"agent_name": "slow", "input": "x", "description": "dig"}),
    )
    .await;
    assert_eq!(
        cond.call(&loop_ctx).await.unwrap(),
        LoopDecision::run_again()
    );
    let reminder = message.call(&loop_ctx).await.unwrap().unwrap();
    assert!(reminder[0].text().contains("- #1 (slow): dig"));

    gate.add_permits(1);
    call(
        &ctx,
        "background_agents_wait_for_first_completion",
        json!({"task_ids": [1]}),
    )
    .await;
    assert_eq!(cond.call(&loop_ctx).await.unwrap(), LoopDecision::stop());
}
