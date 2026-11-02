//! Message injection: messages queued on a session reach the next model
//! call of a run, and a queue left at the end of a turn earns another call.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::prelude::*;
use agent_framework_core::session::{enqueue_messages, pending_messages};
use async_trait::async_trait;

/// A one-shot action run during a model call.
type Action = Box<dyn FnOnce() + Send>;

/// Replies from a script, records each request's texts, and runs an action
/// (such as queueing a message) during the first call.
#[derive(Clone)]
struct Scripted {
    replies: Arc<Mutex<VecDeque<String>>>,
    seen: Arc<Mutex<Vec<Vec<String>>>>,
    during_first: Arc<Mutex<Option<Action>>>,
}

#[async_trait]
impl ChatClient for Scripted {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        _options: ChatOptions,
    ) -> agent_framework_core::error::Result<ChatResponse> {
        self.seen
            .lock()
            .unwrap()
            .push(messages.iter().map(Message::text).collect());
        if let Some(action) = self.during_first.lock().unwrap().take() {
            action();
        }
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| "done".into());
        Ok(ChatResponse::from_text(reply))
    }

    async fn get_streaming_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> agent_framework_core::error::Result<ChatStream> {
        unimplemented!("not used")
    }
}

fn scripted(replies: &[&str]) -> Scripted {
    Scripted {
        replies: Arc::new(Mutex::new(replies.iter().map(|r| r.to_string()).collect())),
        seen: Arc::default(),
        during_first: Arc::default(),
    }
}

#[tokio::test]
async fn queued_messages_join_the_next_request() {
    let client = scripted(&["first answer"]);
    let agent = Agent::builder(client.clone()).build();
    let mut session = agent.create_session();
    enqueue_messages(&session, "also: be brief");
    assert_eq!(pending_messages(&session).len(), 1);
    let response = agent
        .run(vec![Message::user("hello")], Some(&mut session))
        .await
        .unwrap();
    assert!(pending_messages(&session).is_empty());
    let seen = client.seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0], ["hello", "also: be brief"]);
    let texts: Vec<String> = response.messages.iter().map(Message::text).collect();
    assert_eq!(texts, ["also: be brief", "first answer"]);
}

#[tokio::test]
async fn a_message_queued_mid_turn_gets_its_own_answer() {
    let client = scripted(&["answer one", "answer two"]);
    let agent = Agent::builder(client.clone()).build();
    let mut session = agent.create_session();
    let queue_into = session.clone();
    *client.during_first.lock().unwrap() = Some(Box::new(move || {
        enqueue_messages(&queue_into, "one more thing")
    }));
    let response = agent
        .run(vec![Message::user("hi")], Some(&mut session))
        .await
        .unwrap();
    let seen = client.seen.lock().unwrap().clone();
    assert_eq!(
        seen.len(),
        2,
        "the queued message earned another call: {seen:?}"
    );
    assert_eq!(seen[1], ["hi", "answer one", "one more thing"]);
    let texts: Vec<String> = response.messages.iter().map(Message::text).collect();
    assert_eq!(texts, ["answer one", "one more thing", "answer two"]);

    // History recorded the whole exchange, so the next run sees it.
    agent
        .run(vec![Message::user("next")], Some(&mut session))
        .await
        .unwrap();
    let last = client.seen.lock().unwrap().last().cloned().unwrap();
    assert_eq!(
        last,
        ["hi", "answer one", "one more thing", "answer two", "next"]
    );
}
