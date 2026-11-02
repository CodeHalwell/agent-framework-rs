//! `SequentialBuilder::ensure_trailing_user_turn` (upstream #8779).

use std::sync::{Arc, Mutex};

use agent_framework_core::prelude::*;
use agent_framework_core::types::ChatResponseUpdate;
use async_trait::async_trait;
use futures::StreamExt;

/// A chat client that replies with a fixed text and records every prompt it
/// was sent.
#[derive(Clone)]
struct Recording {
    reply: &'static str,
    prompts: Arc<Mutex<Vec<Vec<Message>>>>,
}

#[async_trait]
impl ChatClient for Recording {
    async fn get_response(&self, messages: Vec<Message>, _o: ChatOptions) -> Result<ChatResponse> {
        self.prompts.lock().unwrap().push(messages);
        Ok(ChatResponse::from_text(self.reply))
    }
    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let r = self.get_response(messages, options).await?;
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

fn agent(name: &str, client: Recording) -> Arc<dyn SupportsAgentRun> {
    Arc::new(Agent::builder(client).name(name).build()) as Arc<dyn SupportsAgentRun>
}

fn recording(reply: &'static str) -> Recording {
    Recording {
        reply,
        prompts: Arc::new(Mutex::new(Vec::new())),
    }
}

fn final_conversation(run: &WorkflowRun) -> Vec<Message> {
    let out = run.outputs().last().cloned().expect("an output");
    serde_json::from_value(out).expect("a conversation")
}

#[tokio::test]
async fn a_participant_handed_an_assistant_ending_conversation_gets_a_continuation_turn() {
    let (first, second) = (recording("draft"), recording("final"));
    let workflow = SequentialBuilder::new()
        .add(agent("writer", first.clone()))
        .add(agent("editor", second.clone()))
        .ensure_trailing_user_turn(true)
        .build()
        .unwrap();
    let run = workflow.run("write something").await.unwrap();

    // The first participant's input already ends on the user's turn.
    let first_prompt = first.prompts.lock().unwrap()[0].clone();
    assert_eq!(first_prompt.last().unwrap().text(), "write something");
    // The second is handed the writer's reply, so it gets the continuation.
    let second_prompt = second.prompts.lock().unwrap()[0].clone();
    let last = second_prompt.last().unwrap();
    assert_eq!(last.role, Role::user());
    assert_eq!(last.text(), "please continue.");

    // Invocation-only: it never enters the conversation passed on.
    let conv = final_conversation(&run);
    assert!(
        conv.iter().all(|m| m.text() != "please continue."),
        "the continuation turn leaked into the output: {conv:?}"
    );
    assert_eq!(conv.last().unwrap().text(), "final");
}

#[tokio::test]
async fn off_by_default_the_conversation_is_passed_on_unchanged() {
    let (first, second) = (recording("draft"), recording("final"));
    let workflow = SequentialBuilder::new()
        .add(agent("writer", first))
        .add(agent("editor", second.clone()))
        .build()
        .unwrap();
    workflow.run("write something").await.unwrap();

    let second_prompt = second.prompts.lock().unwrap()[0].clone();
    assert_eq!(second_prompt.last().unwrap().role, Role::assistant());
}
