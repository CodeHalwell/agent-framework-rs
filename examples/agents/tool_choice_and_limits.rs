//! Controlling the function-invocation loop: `ToolMode` decides *whether*
//! the model may call a tool, and `FunctionInvocationConfig` bounds what the
//! loop does once it starts calling them.
//!
//! - `ChatOptions::with_tool_choice(ToolMode::…)` sets the wire-level tool
//!   choice: `Auto` (model decides), `required_any` (it must call something),
//!   `required_function("name")` (it must call that one), or `None` (tools
//!   are visible but off-limits).
//! - `FunctionInvokingChatClient::with_config(FunctionInvocationConfig { … })`
//!   caps the loop: `max_iterations` stops a model that keeps calling tools
//!   forever, `terminate_on_unknown_calls` decides what happens when it
//!   invents a tool name, and `include_detailed_errors` controls whether a
//!   failing tool's real message reaches the model or just a generic
//!   "tool execution failed".
//!
//! The loop lives on the *chat client*, not the agent, so this example wires
//! `FunctionInvokingChatClient` directly -- which is also worth seeing on its
//! own: you can use the framework's tool calling without building an `Agent`.
//!
//! Runs fully offline against scripted clients.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example tool_choice_and_limits
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::types::FunctionArguments;
use async_trait::async_trait;
use serde_json::json;

/// A model that reports back the `tool_choice` it was handed, so the wire
/// value is visible without a real provider.
#[derive(Clone)]
struct EchoToolChoice;

#[async_trait]
impl ChatClient for EchoToolChoice {
    async fn get_response(&self, _m: Vec<Message>, options: ChatOptions) -> Result<ChatResponse> {
        let choice = match &options.tool_choice {
            None => "unset (provider default)".to_string(),
            Some(mode) => match mode {
                ToolMode::Auto => "auto -- the model decides".into(),
                ToolMode::Required(None) => "required -- must call some tool".into(),
                ToolMode::Required(Some(name)) => {
                    format!("required -- must call `{name}` specifically")
                }
                ToolMode::None => "none -- tools are off-limits this turn".into(),
            },
        };
        let names: Vec<&str> = options.tools.iter().map(|t| t.name.as_str()).collect();
        Ok(ChatResponse::from_text(format!(
            "tool_choice={choice}; wire name={:?}; tools offered={names:?}",
            options.tool_choice.as_ref().map(ToolMode::as_str)
        )))
    }

    async fn get_streaming_response(
        &self,
        _m: Vec<Message>,
        _o: ChatOptions,
    ) -> Result<ChatStream> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

/// A model that calls `tick` on every single turn and never stops -- the
/// runaway case `max_iterations` exists to bound.
#[derive(Clone)]
struct NeverStops {
    /// Optionally call a tool name that was never registered, to show
    /// `terminate_on_unknown_calls`.
    call_name: &'static str,
    turns: Arc<AtomicUsize>,
}

#[async_trait]
impl ChatClient for NeverStops {
    async fn get_response(&self, _m: Vec<Message>, _o: ChatOptions) -> Result<ChatResponse> {
        let n = self.turns.fetch_add(1, Ordering::SeqCst);
        Ok(ChatResponse {
            messages: vec![Message::with_contents(
                Role::assistant(),
                vec![Content::FunctionCall(FunctionCallContent::new(
                    format!("call-{n}"),
                    self.call_name,
                    Some(FunctionArguments::Raw(json!({}).to_string())),
                ))],
            )],
            finish_reason: Some(FinishReason::tool_calls()),
            ..Default::default()
        })
    }

    async fn get_streaming_response(
        &self,
        _m: Vec<Message>,
        _o: ChatOptions,
    ) -> Result<ChatStream> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

fn tick_tool() -> ToolDefinition {
    FunctionTool::new(
        "tick",
        "Advance a counter by one.",
        json!({ "type": "object", "properties": {} }),
        |_args| async move { Ok(json!({ "ticked": true })) },
    )
    .into_definition()
}

fn exploding_tool() -> ToolDefinition {
    FunctionTool::new(
        "explode",
        "A tool that always fails, with a very specific message.",
        json!({ "type": "object", "properties": {} }),
        |_args| async move {
            Err::<serde_json::Value, _>(Error::Tool(
                "quota exhausted for tenant acme-corp until 14:00 UTC".into(),
            ))
        },
    )
    .into_definition()
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("== 1. ToolMode: whether the model may call a tool at all ==\n");

    let agent = Agent::builder(EchoToolChoice)
        .name("assistant")
        .tool(tick_tool())
        .build();

    for mode in [
        ToolMode::auto(),
        ToolMode::required_any(),
        ToolMode::required_function("tick"),
        ToolMode::None,
    ] {
        let options = AgentRunOptions::new()
            .with_chat_options(ChatOptions::new().with_tool_choice(mode.clone()));
        let response = agent
            .run_with_options(vec![Message::user("hi")], None, options)
            .await?;
        println!("  {}", response.text());
    }

    println!(
        "\n  `required_function` is how you force a specific call -- useful for\n  \
         extraction pipelines where the tool schema *is* the output schema, and\n  \
         a chatty preamble instead of a call would break the caller."
    );

    println!("\n== 2. max_iterations: bounding a runaway tool loop ==\n");

    let turns = Arc::new(AtomicUsize::new(0));
    let model = NeverStops {
        call_name: "tick",
        turns: turns.clone(),
    };
    // `FunctionInvokingChatClient` is the loop itself: hand it a raw client
    // and it executes tool calls and re-prompts until the model stops asking
    // (or the cap below is hit). `Agent::builder` wraps every client in one
    // of these for you; here we build it directly so its config is reachable.
    let looping = FunctionInvokingChatClient::new(model).with_config(FunctionInvocationConfig {
        max_iterations: 3,
        ..FunctionInvocationConfig::default()
    });

    let options = ChatOptions::new().with_tool(tick_tool());
    let response = looping
        .get_response(vec![Message::user("count forever")], options)
        .await?;
    println!(
        "  model asked for a tool on every turn; it was called {} time(s) before \
         the cap stopped the loop.",
        turns.load(Ordering::SeqCst)
    );
    println!("  finish_reason={:?}", response.finish_reason);
    println!(
        "  (default is 40 -- lower it when a tight latency budget matters more \
         than\n  letting the model work a problem through.)"
    );

    println!("\n== 3. terminate_on_unknown_calls: when the model invents a tool ==\n");

    for terminate in [false, true] {
        let turns = Arc::new(AtomicUsize::new(0));
        let model = NeverStops {
            call_name: "no_such_tool",
            turns: turns.clone(),
        };
        let looping =
            FunctionInvokingChatClient::new(model).with_config(FunctionInvocationConfig {
                max_iterations: 3,
                terminate_on_unknown_calls: terminate,
                ..FunctionInvocationConfig::default()
            });
        let options = ChatOptions::new().with_tool(tick_tool());
        let result = looping
            .get_response(vec![Message::user("go")], options)
            .await;
        let outcome = match &result {
            Ok(r) => format!("ok, finish_reason={:?}", r.finish_reason),
            Err(e) => format!("error: {e}"),
        };
        println!(
            "  terminate_on_unknown_calls={terminate:<5} -> {} model call(s), {outcome}",
            turns.load(Ordering::SeqCst)
        );
    }
    println!(
        "\n  false (the default) hands the model an error result for the unknown\n  \
         name and lets it correct itself; true stops the run instead."
    );

    println!("\n== 4. include_detailed_errors: what a failing tool tells the model ==\n");

    for detailed in [false, true] {
        let model = ReportsToolError;
        let looping =
            FunctionInvokingChatClient::new(model).with_config(FunctionInvocationConfig {
                include_detailed_errors: detailed,
                ..FunctionInvocationConfig::default()
            });
        let options = ChatOptions::new().with_tool(exploding_tool());
        let response = looping
            .get_response(vec![Message::user("go")], options)
            .await?;
        println!(
            "  include_detailed_errors={detailed:<5} -> {}",
            response.text()
        );
    }
    println!(
        "\n  Off by default: a tool's error text can carry internal detail you do\n  \
         not want reaching the model (or, through it, the user). Turn it on when\n  \
         you want the model to actually recover from the specific failure."
    );

    Ok(())
}

/// Calls `explode` once, then reports whatever exception text came back.
#[derive(Clone)]
struct ReportsToolError;

#[async_trait]
impl ChatClient for ReportsToolError {
    async fn get_response(&self, messages: Vec<Message>, _o: ChatOptions) -> Result<ChatResponse> {
        let seen = messages
            .iter()
            .flat_map(|m| m.contents.iter())
            .find_map(|c| match c {
                Content::FunctionResult(r) => Some(r.exception.clone().unwrap_or_default()),
                _ => None,
            });
        match seen {
            Some(exception) => Ok(ChatResponse::from_text(format!(
                "the model was told: {exception:?}"
            ))),
            None => Ok(ChatResponse {
                messages: vec![Message::with_contents(
                    Role::assistant(),
                    vec![Content::FunctionCall(FunctionCallContent::new(
                        "call-1",
                        "explode",
                        Some(FunctionArguments::Raw(json!({}).to_string())),
                    ))],
                )],
                finish_reason: Some(FinishReason::tool_calls()),
                ..Default::default()
            }),
        }
    }

    async fn get_streaming_response(
        &self,
        _m: Vec<Message>,
        _o: ChatOptions,
    ) -> Result<ChatStream> {
        Ok(Box::pin(futures::stream::empty()))
    }
}
