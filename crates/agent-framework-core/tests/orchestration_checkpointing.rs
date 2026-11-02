//! The orchestration builders accept checkpoint storage, as upstream's
//! `checkpoint_storage` / `with_checkpointing`, and the built workflow
//! checkpoints to it.

use std::sync::Arc;

use agent_framework_core::prelude::*;
use agent_framework_core::types::ChatResponseUpdate;
use agent_framework_core::workflow::{CheckpointStorage, InMemoryCheckpointStorage};
use async_trait::async_trait;
use futures::StreamExt;

#[derive(Clone)]
struct Fixed(&'static str);

#[async_trait]
impl ChatClient for Fixed {
    async fn get_response(&self, _m: Vec<Message>, _o: ChatOptions) -> Result<ChatResponse> {
        Ok(ChatResponse::from_text(self.0))
    }
    async fn get_streaming_response(&self, m: Vec<Message>, o: ChatOptions) -> Result<ChatStream> {
        let r = self.get_response(m, o).await?;
        Ok(futures::stream::iter(r.messages.into_iter().map(|m| {
            Ok(ChatResponseUpdate {
                contents: m.contents,
                role: Some(m.role),
                ..Default::default()
            })
        }))
        .boxed())
    }
}

fn agent(name: &str) -> Arc<dyn SupportsAgentRun> {
    Arc::new(Agent::builder(Fixed("ok")).name(name).build()) as Arc<dyn SupportsAgentRun>
}

async fn assert_checkpointed(workflow: Workflow, storage: Arc<InMemoryCheckpointStorage>) {
    workflow.run("go").await.unwrap();
    assert!(
        !storage.list(None).await.unwrap().is_empty(),
        "the built workflow checkpoints to the builder's storage"
    );
}

#[tokio::test]
async fn sequential_builds_a_checkpointing_workflow() {
    let storage = Arc::new(InMemoryCheckpointStorage::new());
    let workflow = SequentialBuilder::new()
        .add(agent("a"))
        .add(agent("b"))
        .with_checkpointing(storage.clone())
        .build()
        .unwrap();
    assert_checkpointed(workflow, storage).await;
}

#[tokio::test]
async fn concurrent_builds_a_checkpointing_workflow() {
    let storage = Arc::new(InMemoryCheckpointStorage::new());
    let workflow = ConcurrentBuilder::new()
        .participants(vec![agent("a"), agent("b")])
        .with_checkpointing(storage.clone())
        .build()
        .unwrap();
    assert_checkpointed(workflow, storage).await;
}

#[tokio::test]
async fn group_chat_builds_a_checkpointing_workflow() {
    let storage = Arc::new(InMemoryCheckpointStorage::new());
    let workflow = GroupChatBuilder::new()
        .participant("A", agent("A"))
        .round_robin()
        .max_rounds(1)
        .with_checkpointing(storage.clone())
        .build()
        .unwrap();
    assert_checkpointed(workflow, storage).await;
}
