//! Load a `Agent` from a declarative YAML spec. The loader is
//! provider-agnostic: you register a `ChatClientFactory` closure per provider
//! key ("OpenAI.Chat" here, i.e. `model.provider` + `model.apiType`), and the
//! spec's model options (temperature, ...), instructions, and tools are wired
//! up for you. `${VAR}` / `${VAR:-default}` interpolation works in string
//! fields.
//!
//! Runs offline (a canned client is registered when OPENAI_API_KEY is unset).
//!
//! ```bash
//! cargo run -p agent-framework-examples --example declarative_agent
//! ```

use std::sync::Arc;

use agent_framework::declarative::{ChatClientFactory, DeclarativeLoader};
use agent_framework::prelude::*;
use async_trait::async_trait;
use futures::StreamExt;

/// A tiny offline stand-in for a model, used when OPENAI_API_KEY is unset.
#[derive(Clone)]
struct CannedClient;

#[async_trait]
impl ChatClient for CannedClient {
    async fn get_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        Ok(ChatResponse::from_text(
            "(canned reply -- set OPENAI_API_KEY for a real model) Hi there!",
        ))
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let resp = self.get_response(messages, options).await?;
        let updates = resp.messages.into_iter().map(|m| {
            Ok(ChatResponseUpdate {
                contents: m.contents,
                role: Some(m.role),
                ..Default::default()
            })
        });
        Ok(futures::stream::iter(updates.collect::<Vec<_>>()).boxed())
    }
}

/// The spec: official schema vocabulary (kind/name/instructions/model/...).
const SPEC: &str = r#"
kind: Prompt
name: Assistant
description: A concise, helpful assistant.
instructions: You are a helpful assistant. Keep answers to one sentence.
model:
  id: ${OPENAI_CHAT_MODEL_ID:-gpt-4o-mini}
  provider: OpenAI
  apiType: Chat
  options:
    temperature: 0.7
"#;

#[tokio::main]
async fn main() -> Result<()> {
    // The factory receives the parsed ModelSpec (id, connection, options) and
    // returns the ChatClient for it. Register one per provider key; the
    // loader tries "provider.apiType" first, then "provider", then a default.
    let factory = ChatClientFactory::new().with("OpenAI.Chat", |model| {
        match OpenAIChatCompletionClient::from_env(model.id.as_deref().unwrap_or("gpt-4o-mini")) {
            Ok(client) => Ok(Arc::new(client) as Arc<dyn ChatClient>),
            Err(_) => Ok(Arc::new(CannedClient) as Arc<dyn ChatClient>),
        }
    });

    let loader = DeclarativeLoader::new().with_client_factory(factory);

    // Loaders also take a ToolRegistry (native Rust tools referenced by name
    // in the spec's `tools:` section) and, for `load_workflow`, an
    // AgentRegistry of pre-built agents.
    let agent = loader
        .load_agent(SPEC)
        .map_err(|e| Error::Configuration(e.to_string()))?;

    println!("loaded agent: {}", agent.display_name());
    let response = agent.run_once("Say hello!").await?;
    println!("{}", response.text());

    Ok(())
}
