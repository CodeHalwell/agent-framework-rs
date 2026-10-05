//! The Magentic task ledger belongs to the run, not to the manager
//! (upstream #8581).
//!
//! A `StandardMagenticManager` used to cache the decomposed ledger (facts +
//! plan) on itself. Two things could share one manager — a caller handing the
//! same `Arc` to two builders, and `Workflow::run` taking `&self`, so one
//! workflow can have two runs in flight — and the cache is read at three
//! places where being wrong is expensive:
//!
//! 1. `replan`, which builds its "update these facts" prompt from the
//!    previous ledger. This is the one that corrupts the run's own reasoning
//!    rather than only a display: a run could be told to update the *other*
//!    run's facts.
//! 2. The plan-review request, where a human is asked to approve a plan.
//! 3. The stall-intervention request, which shows a human the facts and plan.
//!
//! None of it is visible in the output — the run keeps going with a plan built
//! from another task's facts — so these tests pin the isolation directly.
//! They also cover the third thing the move fixes: the ledger is now
//! checkpointed with the run, so a resumed run can still replan.

use std::sync::{Arc, Mutex};

use agent_framework_core::prelude::*;
use agent_framework_core::types::ChatResponseUpdate;
use agent_framework_core::workflow::{MagenticContext, MagenticManager, StandardMagenticManager};
use async_trait::async_trait;
use futures::StreamExt;

/// A chat client that replies from a script *and* records the prompt text of
/// every call, so a test can assert which facts a replan was built from.
#[derive(Clone)]
struct RecordingClient {
    replies: Arc<Mutex<Vec<String>>>,
    seen: Arc<Mutex<Vec<String>>>,
}

impl RecordingClient {
    fn new(replies: &[&str]) -> Self {
        Self {
            replies: Arc::new(Mutex::new(
                replies.iter().map(|s| (*s).to_string()).collect(),
            )),
            seen: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Everything the client has been asked since `from`, as one string.
    fn prompts_since(&self, from: usize) -> String {
        self.seen.lock().unwrap()[from..].join("\n")
    }

    fn call_count(&self) -> usize {
        self.seen.lock().unwrap().len()
    }
}

#[async_trait]
impl ChatClient for RecordingClient {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        self.seen.lock().unwrap().push(
            messages
                .iter()
                .map(Message::text)
                .collect::<Vec<_>>()
                .join("\n"),
        );
        let mut replies = self.replies.lock().unwrap();
        Ok(ChatResponse::from_text(if replies.is_empty() {
            "(exhausted)".to_string()
        } else {
            replies.remove(0)
        }))
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

fn manager(client: RecordingClient) -> StandardMagenticManager {
    StandardMagenticManager::new(
        Arc::new(Agent::builder(client).name("mgr").build()) as Arc<dyn SupportsAgentRun>
    )
}

fn context(task: &str) -> MagenticContext {
    MagenticContext::new(
        Message::user(task),
        vec![("coder".into(), "writes code".into())],
    )
}

/// The central guarantee: one manager, two runs, and a replan on the first
/// must be built from *its own* facts.
///
/// Before the ledger moved onto the context this failed in the worst
/// available way — not an error, but a plan silently derived from the other
/// run's facts.
#[tokio::test]
async fn a_shared_manager_replans_each_run_from_its_own_facts() {
    let client = RecordingClient::new(&[
        // Run A's plan.
        "FACTS-FOR-ALPHA",
        "PLAN-FOR-ALPHA",
        // Run B's plan, which used to overwrite A's cache.
        "FACTS-FOR-BETA",
        "PLAN-FOR-BETA",
        // Run A's replan.
        "UPDATED-FACTS-FOR-ALPHA",
        "UPDATED-PLAN-FOR-ALPHA",
    ]);
    let mgr = manager(client.clone());

    let mut alpha = context("alpha task");
    let mut beta = context("beta task");

    mgr.plan(&mut alpha).await.unwrap();
    mgr.plan(&mut beta).await.unwrap();

    // Each run holds its own ledger.
    assert_eq!(
        alpha.task_ledger.as_ref().unwrap().facts.text(),
        "FACTS-FOR-ALPHA"
    );
    assert_eq!(
        beta.task_ledger.as_ref().unwrap().facts.text(),
        "FACTS-FOR-BETA"
    );

    // Now replan alpha, after beta planned. The facts-update prompt must
    // carry alpha's facts.
    let before = client.call_count();
    mgr.replan(&mut alpha).await.unwrap();
    let prompts = client.prompts_since(before);
    assert!(
        prompts.contains("FACTS-FOR-ALPHA"),
        "alpha's replan must update alpha's facts: {prompts}"
    );
    assert!(
        !prompts.contains("FACTS-FOR-BETA"),
        "alpha's replan was built from beta's facts: {prompts}"
    );

    // And the replan updated alpha's ledger, leaving beta's alone.
    assert_eq!(
        alpha.task_ledger.as_ref().unwrap().facts.text(),
        "UPDATED-FACTS-FOR-ALPHA"
    );
    assert_eq!(
        beta.task_ledger.as_ref().unwrap().facts.text(),
        "FACTS-FOR-BETA"
    );
}

/// The ledger is part of the run's serialized state, so a run that was
/// checkpointed and resumed can still replan.
///
/// With the ledger on the manager this was a hard failure rather than a quiet
/// one: a fresh process has a fresh manager, so the first stall after a resume
/// hit `replan() called before plan()`.
#[tokio::test]
async fn a_resumed_run_can_still_replan() {
    let client = RecordingClient::new(&["FACTS-V1", "PLAN-V1", "FACTS-V2", "PLAN-V2"]);
    let mut ctx = context("task");
    manager(client.clone()).plan(&mut ctx).await.unwrap();

    // Round-trip the run's state the way checkpointing does, and bring it
    // back against a manager that has never planned anything.
    let json = serde_json::to_string(&ctx).unwrap();
    let mut resumed: MagenticContext = serde_json::from_str(&json).unwrap();
    assert_eq!(
        resumed.task_ledger.as_ref().unwrap().facts.text(),
        "FACTS-V1"
    );

    let fresh_client = RecordingClient::new(&["FACTS-V2", "PLAN-V2"]);
    let fresh = manager(fresh_client.clone());
    fresh.replan(&mut resumed).await.unwrap();
    assert!(
        fresh_client.prompts_since(0).contains("FACTS-V1"),
        "the resumed replan must update the facts the run had before the restart"
    );
    assert_eq!(
        resumed.task_ledger.as_ref().unwrap().facts.text(),
        "FACTS-V2"
    );
}

/// A reset precedes every replan, and the replan's job is to *update* the
/// previous facts — so the reset must not clear them.
#[tokio::test]
async fn a_reset_keeps_the_ledger_the_following_replan_needs() {
    let client = RecordingClient::new(&["FACTS-V1", "PLAN-V1"]);
    let mut ctx = context("task");
    manager(client).plan(&mut ctx).await.unwrap();
    ctx.chat_history.push(Message::assistant("some work"));

    ctx.reset();

    // The things a reset does clear.
    assert!(ctx.chat_history.is_empty());
    assert_eq!(ctx.stall_count, 0);
    assert_eq!(ctx.reset_count, 1);
    // And the one it must not.
    assert_eq!(ctx.task_ledger.as_ref().unwrap().facts.text(), "FACTS-V1");
}

/// Replanning before planning is still an error rather than a silent
/// fresh plan: a replan with no previous facts is not a replan.
#[tokio::test]
async fn replan_without_a_prior_plan_is_still_an_error() {
    let client = RecordingClient::new(&["unused"]);
    let mut ctx = context("task");
    let err = manager(client)
        .replan(&mut ctx)
        .await
        .expect_err("replan before plan");
    assert!(err.to_string().contains("replan() called before plan()"));
}
