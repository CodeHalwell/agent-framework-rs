//! Smaller parity options: todo list rendering, the all-tools approval rule,
//! skills provider options, response-to-updates and message sources.

use std::sync::Arc;

use agent_framework_core::memory::{
    message_source_id, message_source_type, with_message_source, ContextProvider, SessionContext,
};
use agent_framework_core::prelude::*;
use agent_framework_core::skills::{Skill, SkillsProvider};

#[cfg(feature = "experimental-harness")]
#[tokio::test]
async fn the_todo_list_message_can_be_suppressed_or_rebuilt() {
    use agent_framework_core::harness::TodoProvider;
    let session = AgentSession::new();
    let context = || {
        let mut ctx = SessionContext::new(vec![]);
        ctx.session = Some(session.clone());
        ctx
    };
    let mut ctx = context();
    TodoProvider::new()
        .with_suppressed_todo_list_message(true)
        .before_run(&mut ctx)
        .await
        .unwrap();
    assert!(ctx.messages.is_empty());
    let mut ctx = context();
    TodoProvider::new()
        .with_todo_list_message_builder(|items| format!("{} todos", items.len()))
        .before_run(&mut ctx)
        .await
        .unwrap();
    assert_eq!(ctx.messages[0].text(), "0 todos");
}

#[cfg(feature = "experimental-harness")]
#[test]
fn the_all_tools_rule_approves_everything() {
    use agent_framework_core::harness::all_tools_auto_approval_rule;
    use agent_framework_core::types::FunctionCallContent;
    assert!(all_tools_auto_approval_rule(&FunctionCallContent::new(
        "c", "anything", None
    )));
}

#[tokio::test]
async fn skills_provider_options_shape_the_catalog_and_tools() {
    let provider = SkillsProvider::new(vec![
        Skill::new("keep", "Kept."),
        Skill::new("drop", "Dropped."),
    ])
    .with_filter(|s| s.name == "keep")
    .with_instructions_prompt("Skills you can use:\n{skills}")
    .with_load_skill_approval(true);
    let mut ctx = SessionContext::new(vec![]);
    provider.before_run(&mut ctx).await.unwrap();
    let instructions = ctx.instructions.unwrap();
    assert!(instructions.starts_with("Skills you can use:\n- keep: Kept."));
    assert!(!instructions.contains("drop"));
    let tool = |name: &str| ctx.tools.iter().find(|t| t.name == name).unwrap();
    assert!(tool("load_skill").requires_approval());
    assert!(!tool("read_skill_resource").requires_approval());
}

#[test]
fn a_response_round_trips_through_its_updates() {
    let response = AgentResponse {
        messages: vec![Message::assistant("one"), Message::assistant("two")],
        response_id: Some("r1".into()),
        ..Default::default()
    };
    let updates = response.to_updates();
    assert_eq!(updates.len(), 2);
    let back = AgentResponse::from_updates(updates);
    let texts: Vec<String> = back.messages.iter().map(Message::text).collect();
    assert_eq!(texts, ["one", "two"]);
    assert_eq!(back.response_id.as_deref(), Some("r1"));
}

#[test]
fn messages_carry_a_source_type_and_id() {
    let m = with_message_source(Message::user("from outside"), "External", "webhook");
    assert_eq!(message_source_type(&m), Some("External"));
    assert_eq!(message_source_id(&m), Some("webhook"));
    assert_eq!(message_source_type(&Message::user("x")), None);
}

#[test]
fn an_agent_exposes_its_chat_client() {
    #[derive(Clone)]
    struct Model;
    #[async_trait::async_trait]
    impl agent_framework_core::client::ChatClient for Model {
        async fn get_response(
            &self,
            _: Vec<Message>,
            _: ChatOptions,
        ) -> agent_framework_core::error::Result<ChatResponse> {
            Ok(ChatResponse::from_text("ok"))
        }
        async fn get_streaming_response(
            &self,
            _: Vec<Message>,
            _: ChatOptions,
        ) -> agent_framework_core::error::Result<agent_framework_core::client::ChatStream> {
            unimplemented!("not used")
        }
        fn model(&self) -> Option<&str> {
            Some("m-1")
        }
    }
    let agent = Agent::builder(Model).build();
    assert_eq!(agent.chat_client().model(), Some("m-1"));
    let _: &Arc<dyn agent_framework_core::client::ChatClient> = agent.chat_client();
}
