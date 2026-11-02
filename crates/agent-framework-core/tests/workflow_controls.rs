//! Workflow run controls: halting, cancelling, run ids, executor reset,
//! warnings, and the orchestration options for chaining, aggregation and
//! reassigning other agents' turns.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::error::Result;
use agent_framework_core::prelude::*;
use agent_framework_core::workflow::{
    ConcurrentBuilder, Executor, FunctionExecutor, SequentialBuilder, WorkflowBuilder,
    WorkflowContext, WorkflowEvent, WorkflowExecutor,
};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

#[tokio::test]
async fn request_halt_stops_the_run_after_the_superstep() {
    let reached_b = Arc::new(AtomicUsize::new(0));
    let counter = reached_b.clone();
    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(FunctionExecutor::new(
            "a",
            |msg, ctx| async move {
                ctx.send_message(msg).await?;
                ctx.yield_output(json!("a done")).await?;
                ctx.request_halt();
                Ok(())
            },
        )))
        .add_executor(Arc::new(FunctionExecutor::new("b", move |_msg, _ctx| {
            let counter = counter.clone();
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        })))
        .set_start("a")
        .add_edge("a", "b")
        .build()
        .unwrap();
    let run = workflow.run(json!("go")).await.unwrap();
    assert_eq!(reached_b.load(Ordering::SeqCst), 0, "b never ran");
    assert_eq!(run.outputs(), [json!("a done")], "a's effects were applied");
    let other = workflow.run(json!("go")).await.unwrap();
    assert_ne!(run.session_id(), other.session_id());
}

#[tokio::test]
async fn a_streamed_run_can_be_cancelled() {
    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(FunctionExecutor::new(
            "slow",
            |_msg, _ctx| async move {
                tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                Ok(())
            },
        )))
        .set_start("slow")
        .build()
        .unwrap();
    let mut stream = workflow.run_stream(json!("go"));
    assert!(!stream.session_id().is_empty());
    // Let the run start, then cancel it; the stream ends promptly.
    let _ = stream.next().await;
    stream.cancel();
    let drained = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while stream.next().await.is_some() {}
    })
    .await;
    assert!(drained.is_ok(), "the cancelled stream ended");
    assert!(stream.into_run().await.is_err(), "cancellation is reported");
}

/// Counts its invocations; the count is per run, cleared by `reset`.
struct Counting {
    seen: AtomicUsize,
    resets: AtomicUsize,
}

#[async_trait]
impl Executor for Counting {
    fn id(&self) -> &str {
        "count"
    }
    async fn execute(&self, _message: Value, ctx: WorkflowContext) -> Result<()> {
        let n = self.seen.fetch_add(1, Ordering::SeqCst) + 1;
        ctx.yield_output(json!(n)).await
    }
    async fn reset(&self) -> Result<()> {
        self.seen.store(0, Ordering::SeqCst);
        self.resets.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[tokio::test]
async fn executors_are_reset_before_each_run() {
    let counting = Arc::new(Counting {
        seen: AtomicUsize::new(0),
        resets: AtomicUsize::new(0),
    });
    let workflow = WorkflowBuilder::new()
        .add_executor(counting.clone())
        .set_start("count")
        .build()
        .unwrap();
    for _ in 0..2 {
        let run = workflow.run(json!(null)).await.unwrap();
        assert_eq!(run.outputs(), [json!(1)], "each run starts fresh");
    }
    assert_eq!(counting.resets.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn warnings_surface_and_nested_warnings_are_attributed() {
    let child = WorkflowBuilder::new()
        .add_executor(Arc::new(FunctionExecutor::new(
            "inner",
            |msg, ctx| async move {
                ctx.add_warning("inner is slow");
                ctx.yield_output(msg).await
            },
        )))
        .set_start("inner")
        .build()
        .unwrap();
    let child_id = child.id().to_string();
    let parent = WorkflowBuilder::new()
        .add_executor(Arc::new(WorkflowExecutor::new("sub", child)))
        .set_start("sub")
        .build()
        .unwrap();
    let run = parent.run(json!("x")).await.unwrap();
    let forwarded: Vec<_> = run
        .events()
        .iter()
        .filter_map(|e| match e {
            WorkflowEvent::SubworkflowWarning {
                sub_workflow_id,
                message,
            } => Some((sub_workflow_id.clone(), message.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(forwarded, [(child_id, "inner is slow".to_string())]);
}

/// Each request's `(role, text)` pairs.
type Seen = Arc<Mutex<Vec<Vec<(String, String)>>>>;

/// Replies with its name and records what it was sent.
#[derive(Clone)]
struct Named {
    name: String,
    seen: Seen,
}

#[async_trait]
impl ChatClient for Named {
    async fn get_response(&self, messages: Vec<Message>, _: ChatOptions) -> Result<ChatResponse> {
        self.seen.lock().unwrap().push(
            messages
                .iter()
                .map(|m| (m.role.to_string(), m.text()))
                .collect(),
        );
        Ok(ChatResponse::from_text(format!("from {}", self.name)))
    }
    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let response = self.get_response(messages, options).await?;
        let update = ChatResponseUpdate {
            contents: vec![Content::text(response.text())],
            role: Some(Role::assistant()),
            ..Default::default()
        };
        Ok(futures::stream::iter(vec![Ok(update)]).boxed())
    }
}

fn agent(
    name: &str,
) -> (
    Arc<dyn SupportsAgentRun>,
    Arc<Mutex<Vec<Vec<(String, String)>>>>,
) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let client = Named {
        name: name.into(),
        seen: seen.clone(),
    };
    (Arc::new(Agent::builder(client).name(name).build()), seen)
}

#[tokio::test]
async fn sequential_can_chain_only_agent_responses() {
    let (a, _) = agent("a");
    let (b, b_seen) = agent("b");
    let workflow = SequentialBuilder::new()
        .participants(vec![a, b])
        .with_chain_only_agent_responses(true)
        .build()
        .unwrap();
    workflow.run(json!("hello")).await.unwrap();
    let texts: Vec<String> = b_seen.lock().unwrap()[0]
        .iter()
        .map(|(_, t)| t.clone())
        .collect();
    assert_eq!(texts, ["from a"], "b sees only a's reply");
}

#[tokio::test]
async fn concurrent_takes_a_custom_aggregator() {
    let (a, _) = agent("a");
    let (b, _) = agent("b");
    let workflow = ConcurrentBuilder::new()
        .participants(vec![a, b])
        .with_aggregator(|conversations| {
            let joined = conversations
                .iter()
                .filter_map(|c| c.last())
                .map(Message::text)
                .collect::<Vec<_>>()
                .join(" + ");
            vec![Message::assistant(joined)]
        })
        .build()
        .unwrap();
    let run = workflow.run(json!("hello")).await.unwrap();
    let output = run.last_output().unwrap();
    let messages: Vec<Message> = serde_json::from_value(output).unwrap();
    assert_eq!(messages[0].text(), "from a + from b");
}

#[tokio::test]
async fn other_agents_can_be_presented_as_users() {
    use agent_framework_core::workflow::AgentExecutor;
    let (b, b_seen) = agent("b");
    let mut from_a = Message::assistant("from a");
    from_a.author_name = Some("a".into());
    let conversation = serde_json::to_value(vec![Message::user("hello"), from_a]).unwrap();
    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(
            AgentExecutor::new("b", b)
                .with_output(true)
                .with_reassign_other_agents_as_users(true),
        ))
        .set_start("b")
        .build()
        .unwrap();
    workflow.run(conversation).await.unwrap();
    let roles: Vec<String> = b_seen.lock().unwrap()[0]
        .iter()
        .map(|(r, _)| r.clone())
        .collect();
    assert_eq!(roles, ["user", "user"]);
}
