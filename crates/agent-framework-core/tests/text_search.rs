//! `TextSearchProvider` driven through a real `Agent`, after upstream's
//! .NET `TextSearchProviderTests` and Go's `textsearch` provider tests.

use std::sync::{Arc, Mutex};

use agent_framework_core::prelude::*;
use agent_framework_core::text_search::{
    TextSearchBehavior, TextSearchProvider, TextSearchResult, DEFAULT_TEXT_SEARCH_STATE_KEY,
};
use agent_framework_core::types::Role;
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::json;

/// Each request a client was sent, with its options.
type Seen = Arc<Mutex<Vec<(Vec<Message>, ChatOptions)>>>;

/// Answers "ok" and records what it was sent; fails any request whose last
/// message says "boom".
#[derive(Clone, Default)]
struct RecordingClient {
    seen: Seen,
}

#[async_trait]
impl ChatClient for RecordingClient {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        let fail = messages.last().is_some_and(|m| m.text() == "boom");
        self.seen.lock().unwrap().push((messages, options));
        if fail {
            return Err(Error::AgentExecution("scripted failure".into()));
        }
        Ok(ChatResponse::from_text("ok"))
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

/// A search that records each query and returns one fixed hit.
fn recording_search(queries: Arc<Mutex<Vec<String>>>) -> TextSearchProvider {
    TextSearchProvider::new(move |query: String| {
        let queries = queries.clone();
        async move {
            queries.lock().unwrap().push(query);
            Ok(vec![TextSearchResult::new(
                "Returns are accepted within 30 days.",
            )
            .with_source("Returns policy", "https://example.com/returns")])
        }
    })
}

#[tokio::test]
async fn results_are_added_ahead_of_the_input_before_the_run() {
    let queries = Arc::new(Mutex::new(Vec::new()));
    let client = RecordingClient::default();
    let agent = Agent::builder(client.clone())
        .context_provider(Arc::new(recording_search(queries.clone())))
        .build();

    agent.run_once("What is the returns policy?").await.unwrap();

    assert_eq!(*queries.lock().unwrap(), ["What is the returns policy?"]);
    let (messages, _) = client.seen.lock().unwrap()[0].clone();
    let context = messages
        .iter()
        .find(|m| m.text().starts_with("## Additional Context"))
        .expect("search results were not sent to the model");
    assert_eq!(context.role, Role::user());
    assert!(context.text().contains("SourceDocName: Returns policy"));
    assert!(context
        .text()
        .contains("Contents: Returns are accepted within 30 days."));
    assert_eq!(
        messages.last().unwrap().text(),
        "What is the returns policy?"
    );
}

#[tokio::test]
async fn on_demand_offers_a_search_tool_instead_of_searching() {
    let queries = Arc::new(Mutex::new(Vec::new()));
    let client = RecordingClient::default();
    let provider = recording_search(queries.clone())
        .with_behavior(TextSearchBehavior::OnDemandFunctionCalling)
        .with_tool("SearchPolicies", "Searches the policy documents.");
    let agent = Agent::builder(client.clone())
        .context_provider(Arc::new(provider))
        .build();

    agent.run_once("What is the returns policy?").await.unwrap();

    assert!(queries.lock().unwrap().is_empty());
    let (messages, options) = client.seen.lock().unwrap()[0].clone();
    assert_eq!(messages.len(), 1);
    let names: Vec<_> = options.tools.iter().map(|t| t.name.clone()).collect();
    assert_eq!(names, ["SearchPolicies"]);
}

#[tokio::test]
async fn the_search_tool_formats_its_results() {
    let queries = Arc::new(Mutex::new(Vec::new()));
    let provider = recording_search(queries.clone());
    let tool = provider.search_tool();
    let output = tool
        .executor
        .expect("the search tool is executable")
        .invoke(json!({ "userQuestion": "returns" }))
        .await
        .unwrap();
    assert_eq!(*queries.lock().unwrap(), ["returns"]);
    assert!(output
        .to_string()
        .contains("SourceDocLink: https://example.com/returns"));
}

#[tokio::test]
async fn a_failed_search_does_not_fail_the_run() {
    let client = RecordingClient::default();
    let provider = TextSearchProvider::new(|_query: String| async {
        Err(Error::AgentExecution("index offline".into()))
    });
    let agent = Agent::builder(client.clone())
        .context_provider(Arc::new(provider))
        .build();

    let response = agent.run_once("hello").await.unwrap();

    assert_eq!(response.text(), "ok");
    let (messages, _) = client.seen.lock().unwrap()[0].clone();
    assert_eq!(messages.len(), 1);
}

#[tokio::test]
async fn no_results_add_no_message() {
    let client = RecordingClient::default();
    let provider = TextSearchProvider::new(|_query: String| async { Ok(vec![]) });
    let agent = Agent::builder(client.clone())
        .context_provider(Arc::new(provider))
        .build();

    agent.run_once("hello").await.unwrap();

    let (messages, _) = client.seen.lock().unwrap()[0].clone();
    assert_eq!(messages.len(), 1);
}

#[tokio::test]
async fn recent_messages_are_searched_with_the_follow_up() {
    let queries = Arc::new(Mutex::new(Vec::new()));
    let provider = recording_search(queries.clone())
        .with_recent_message_memory(2)
        .with_recent_message_roles(vec![Role::user(), Role::assistant()]);
    let agent = Agent::builder(RecordingClient::default())
        .context_provider(Arc::new(provider))
        .build();
    let mut session = agent.create_session();

    agent
        .run(vec![Message::user("first")], Some(&mut session))
        .await
        .unwrap();
    agent
        .run(vec![Message::user("second")], Some(&mut session))
        .await
        .unwrap();
    agent
        .run(vec![Message::user("third")], Some(&mut session))
        .await
        .unwrap();

    let queries = queries.lock().unwrap().clone();
    assert_eq!(queries[0], "first");
    assert_eq!(queries[1], "first\nok\nsecond");
    // Only the last two remembered messages are kept.
    assert_eq!(queries[2], "second\nok\nthird");
    assert_eq!(
        session.state.get(DEFAULT_TEXT_SEARCH_STATE_KEY),
        Some(json!({ "recentMessagesText": ["third", "ok"] }))
    );
}

#[tokio::test]
async fn memory_is_kept_per_session_and_holds_user_messages_by_default() {
    let queries = Arc::new(Mutex::new(Vec::new()));
    let provider = recording_search(queries.clone()).with_recent_message_memory(5);
    let agent = Agent::builder(RecordingClient::default())
        .context_provider(Arc::new(provider))
        .build();
    let mut one = agent.create_session();
    let mut two = agent.create_session();

    agent
        .run(vec![Message::user("alpha")], Some(&mut one))
        .await
        .unwrap();
    agent
        .run(vec![Message::user("beta")], Some(&mut two))
        .await
        .unwrap();
    agent
        .run(vec![Message::user("gamma")], Some(&mut one))
        .await
        .unwrap();

    let queries = queries.lock().unwrap().clone();
    assert_eq!(queries, ["alpha", "beta", "alpha\ngamma"]);
}

#[tokio::test]
async fn a_failed_run_is_not_remembered() {
    let queries = Arc::new(Mutex::new(Vec::new()));
    let provider = recording_search(queries.clone()).with_recent_message_memory(5);
    let agent = Agent::builder(RecordingClient::default())
        .context_provider(Arc::new(provider))
        .build();
    let mut session = agent.create_session();

    assert!(agent
        .run(vec![Message::user("boom")], Some(&mut session))
        .await
        .is_err());
    agent
        .run(vec![Message::user("after")], Some(&mut session))
        .await
        .unwrap();

    assert_eq!(*queries.lock().unwrap(), ["boom", "after"]);
}

#[tokio::test]
async fn a_custom_formatter_replaces_the_default() {
    let client = RecordingClient::default();
    let provider = TextSearchProvider::new(|_query: String| async {
        Ok(vec![TextSearchResult::new("a"), TextSearchResult::new("b")])
    })
    .with_context_formatter(|results| {
        Ok(format!(
            "hits: {}",
            results
                .iter()
                .map(|r| r.text.as_str())
                .collect::<Vec<_>>()
                .join(",")
        ))
    });
    let agent = Agent::builder(client.clone())
        .context_provider(Arc::new(provider))
        .build();

    agent.run_once("q").await.unwrap();

    let (messages, _) = client.seen.lock().unwrap()[0].clone();
    assert!(messages.iter().any(|m| m.text() == "hits: a,b"));
}
