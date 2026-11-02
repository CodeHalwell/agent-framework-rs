//! Context providers see the agent they run for.

use std::sync::{Arc, Mutex};

use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::error::Result;
use agent_framework_core::memory::{AgentInfo, ContextProvider, SessionContext};
use agent_framework_core::prelude::*;
use async_trait::async_trait;

struct Echo;

#[async_trait]
impl ChatClient for Echo {
    async fn get_response(&self, _: Vec<Message>, _: ChatOptions) -> Result<ChatResponse> {
        Ok(ChatResponse::from_text("ok"))
    }
    async fn get_streaming_response(&self, _: Vec<Message>, _: ChatOptions) -> Result<ChatStream> {
        unimplemented!("not used")
    }
}

#[derive(Default)]
struct Spy(Mutex<Option<AgentInfo>>);

#[async_trait]
impl ContextProvider for Spy {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        *self.0.lock().unwrap() = ctx.agent.clone();
        Ok(())
    }
}

#[tokio::test]
async fn before_run_sees_the_agent_identity() {
    let spy = Arc::new(Spy::default());
    let agent = Agent::builder(Echo)
        .id("agent-7")
        .name("planner")
        .description("plans things")
        .context_provider(spy.clone())
        .build();
    agent.run_once("hi").await.unwrap();
    assert_eq!(
        spy.0.lock().unwrap().clone(),
        Some(AgentInfo {
            id: "agent-7".into(),
            name: Some("planner".into()),
            description: Some("plans things".into()),
        })
    );
}

struct Facts;

#[async_trait]
impl ContextProvider for Facts {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        ctx.messages.push(Message::system("the sky is blue"));
        ctx.extend_messages_from(
            "memory",
            None,
            vec![Message::system("you met Ada")],
            &["other-session".to_string()],
        );
        Ok(())
    }
}

#[derive(Clone, Default)]
struct Seen(Arc<Mutex<Vec<Message>>>);

#[async_trait]
impl ChatClient for Seen {
    async fn get_response(&self, messages: Vec<Message>, _: ChatOptions) -> Result<ChatResponse> {
        *self.0.lock().unwrap() = messages;
        Ok(ChatResponse::from_text("ok"))
    }
    async fn get_streaming_response(&self, _: Vec<Message>, _: ChatOptions) -> Result<ChatStream> {
        unimplemented!("not used")
    }
}

#[tokio::test]
async fn context_messages_carry_their_source() {
    use agent_framework_core::memory::{message_source_id, ATTRIBUTION_KEY};
    let client = Seen::default();
    let agent = Agent::builder(client.clone())
        .context_provider(Arc::new(Facts))
        .build();
    agent.run_once("hi").await.unwrap();
    let seen = client.0.lock().unwrap().clone();
    let sources: Vec<Option<&str>> = seen.iter().map(message_source_id).collect();
    assert_eq!(sources, [Some("Facts"), Some("memory"), None]);
    assert_eq!(
        seen[1].additional_properties[ATTRIBUTION_KEY]["origin_session_ids"],
        serde_json::json!(["other-session"])
    );
    assert_eq!(
        seen[0].additional_properties[ATTRIBUTION_KEY]["source_type"],
        "Facts"
    );
    assert_eq!(seen[2].text(), "hi", "the run's input is not attributed");
}
